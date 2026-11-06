"""Whole-set marker readout: one sequence carries every option.

The per-option marker readout isolates each candidate in its own row, so an
option never sees the others.  This readout instead concatenates the options
back to back in a single suffix::

    [context] [option 1] <m> [option 2] <m> ... [option K] <m>

Each marker reads hidden state that has attended to the context and to every
option before it, and the K marker logits are normalised across the set.  It
is the readout a listwise reranker uses, and it also scores a shortlist: pass
a subset and the same head returns a distribution over that subset.

The prefix pass is shared across calls through the KV cache; on a hybrid
(GDN) backbone each call runs on a fork of the one-row prefix cache, because
the recurrent state cannot be cropped.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .decision_model import DecisionModel
from .hybrid_cache import fork_prefix_cache
from .shape_buckets import PREFIX_BUCKETS, SUFFIX_BUCKETS, bucket_length, left_pad


class ListwiseMarkerReadout:
    """Score a set of options with one sequence and the marker head.

    Args:
        model: A :class:`DecisionModel` built with ``option_marker=True``; the
            readout reuses its tokenizer, backbone, layer readout and marker
            head, and adds no parameters of its own.
    """

    def __init__(self, model: DecisionModel) -> None:
        if not model.option_marker:
            raise ValueError("ListwiseMarkerReadout needs option_marker=True")
        self.model = model

    # -- layout ---------------------------------------------------------
    def _marker_hidden(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
    ) -> torch.Tensor:
        """Hidden state at every option marker, [K, hidden].

        The options share one suffix sequence, so an option sees its
        predecessors.  The prefix has to leave room for the whole suffix, not
        for one option.
        """
        model = self.model
        option_ids = model._tokenize_option_ids(list(option_texts))
        blocks = [ids + [model.marker_token_id] for ids in option_ids]
        suffix_len = sum(len(block) for block in blocks)
        if suffix_len >= model.max_length:
            raise ValueError(
                f"listwise readout needs room for {suffix_len} suffix tokens "
                f"inside max_length={model.max_length}; shorten the state, the "
                "options, or the option set"
            )
        # Fixed shapes: every item otherwise presents a new (prefix, suffix)
        # pair, and a new tensor shape costs a 0.4-1.3 s first kernel load on
        # H100.  Bucketing turned a measured 0.78 items/s into the steady-state
        # rate this readout is meant to have.  Padding is masked out, so the
        # real tokens see the same computation.
        width = bucket_length(
            suffix_len, SUFFIX_BUCKETS, limit=model.max_length
        )
        prefix_inputs = model._tokenize_context(
            state, instructions, model.max_length - width
        )
        real_prefix_ids = prefix_inputs["input_ids"]
        real_prefix_mask = prefix_inputs["attention_mask"]
        real_prefix_len = real_prefix_ids.shape[1]
        prefix_bucket = bucket_length(
            real_prefix_len, PREFIX_BUCKETS, limit=model.max_length - width
        )
        if prefix_bucket + width > model.max_length:
            prefix_bucket, width = real_prefix_len, suffix_len
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(model.tokenizer, "eos_token_id", None) or 0
        padded_ids, pad = left_pad(
            prefix_inputs["input_ids"][0].tolist(), prefix_bucket, pad_id
        )
        prefix_inputs = {
            "input_ids": torch.tensor(
                [padded_ids], dtype=torch.long, device=model.device
            ),
            "attention_mask": torch.cat(
                [
                    torch.zeros((1, pad), dtype=torch.long, device=model.device),
                    torch.ones(
                        (1, real_prefix_len), dtype=torch.long, device=model.device
                    ),
                ],
                dim=1,
            ),
            "position_ids": torch.cat(
                [
                    torch.zeros((1, pad), dtype=torch.long, device=model.device),
                    torch.arange(real_prefix_len, device=model.device).unsqueeze(0),
                ],
                dim=1,
            ),
        }

        ids: list[int] = []
        positions: list[int] = []
        for block in blocks:
            ids.extend(block)
            positions.append(len(ids) - 1)
        suffix = torch.zeros((1, width), dtype=torch.long, device=model.device)
        suffix[0, :suffix_len] = torch.tensor(
            ids, dtype=torch.long, device=model.device
        )
        marker_index = torch.tensor(
            positions, dtype=torch.long, device=model.device
        )
        suffix_mask = torch.zeros(
            (1, width), dtype=torch.long, device=model.device
        )
        suffix_mask[0, :suffix_len] = 1
        attention_mask = torch.cat(
            [prefix_inputs["attention_mask"], suffix_mask], dim=1
        )

        prefix_outputs = model._backbone_forward(
            prefix_inputs, output_hidden_states=False, use_cache=True
        )
        cache = getattr(prefix_outputs, "past_key_values", None)
        if cache is not None and model.is_hybrid and model.backbone_requires_grad:
            # A grad-enabled GDN backbone cannot share a prefix cache: the
            # suffix forward overwrites the fork's conv/recurrent states in
            # place, which bumps the version of tensors the prefix graph saved
            # for backward (``RuntimeError: one of the variables needed for
            # gradient computation has been modified by an inplace operation``
            # in fla's chunked delta-rule backward).  Re-encode the context so
            # one autograd graph covers prefix and suffix.
            if not getattr(model, "_warned_lora_hybrid_cache", False):
                print(
                    "  LoRA + hybrid backbone: re-encoding the context instead "
                    "of reusing the GDN cache",
                    flush=True,
                )
                model._warned_lora_hybrid_cache = True
            cache = None
        if cache is None:
            # No KV cache (checkpointing, encoder, adapted hybrid): pay the
            # context again.  Use the unpadded prefix: a left-padded query at
            # position 0 has every key masked, and cuDNN's SDPA backward
            # returns NaN for that row under autocast.
            input_ids = torch.cat(
                [real_prefix_ids, suffix[:, :suffix_len]], dim=1
            )
            attention_mask = torch.cat(
                [real_prefix_mask, suffix_mask[:, :suffix_len]], dim=1
            )
            outputs = model._backbone_forward(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            marker_index = marker_index + real_prefix_len
        else:
            position_ids = torch.full(
                (1, width), real_prefix_len, dtype=torch.long, device=model.device
            )
            position_ids[0, :suffix_len] = torch.arange(
                real_prefix_len,
                real_prefix_len + suffix_len,
                device=model.device,
            )
            chunk_cache = fork_prefix_cache(cache, 1) if model.is_hybrid else cache
            outputs = model._backbone_forward(
                {
                    "input_ids": suffix,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": chunk_cache,
                }
            )
            if not model.is_hybrid:
                # The suffix K/V stay in the cache; drop them so a second call
                # starts from the same prefix.
                model._crop_cache(cache, suffix.shape[1])

        hidden = model.projector(
            model.readout_project(model.readout(outputs.hidden_states).float())
        )
        return hidden[0, marker_index]

    # -- capacity -------------------------------------------------------
    def suffix_length(self, option_texts: Sequence[str]) -> int:
        """Tokens in the option suffix, markers included."""
        return sum(
            len(ids) + 1
            for ids in self.model._tokenize_option_ids(list(option_texts))
        )

    def fits(self, option_texts: Sequence[str]) -> bool:
        """Whether the whole set fits one sequence under ``max_length``."""
        return self.suffix_length(option_texts) < self.model.max_length

    # -- scoring --------------------------------------------------------
    def score(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
    ) -> torch.Tensor:
        """Marker logits [1, K] for ``option_texts``, in the given order."""
        if not option_texts:
            raise ValueError("listwise readout needs at least one option")
        hidden = self._marker_hidden(state, instructions, option_texts)
        return self.model.marker_head(hidden.unsqueeze(0))

    def probabilities(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
    ) -> torch.Tensor:
        """Softmax over the set, [1, K]."""
        return torch.softmax(self.score(state, instructions, option_texts), dim=-1)

    def score_shifted(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
        shifts: int = 4,
    ) -> torch.Tensor:
        """Average the set distribution over cyclic shifts of the order.

        The head is trained on shuffled orders, so averaging a few shifts
        removes part of the order variance.  ``shifts`` is clamped to the
        option count, and for K <= 4 every shift is used.
        """
        n_options = len(option_texts)
        shifts = max(1, min(shifts, n_options))
        total: torch.Tensor | None = None
        for shift in range(shifts):
            order = [index % n_options for index in range(shift, shift + n_options)]
            rotated = [option_texts[index] for index in order]
            probs = self.probabilities(state, instructions, rotated)
            unrotated = torch.empty_like(probs)
            for position, source in enumerate(order):
                unrotated[0, source] = probs[0, position]
            total = unrotated if total is None else total + unrotated
        assert total is not None
        return total / shifts

    def set_loss(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
        gold_index: int,
    ) -> torch.Tensor:
        """Cross entropy over the set for the option at ``gold_index``."""
        logits = self.score(state, instructions, option_texts)
        target = torch.tensor([gold_index], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    # -- training order -------------------------------------------------
    @staticmethod
    def sample_subset(
        n_options: int,
        gold_index: int,
        rng: random.Random,
        min_size: int = 2,
        max_size: int | None = None,
    ) -> list[int]:
        """A shuffled subset that contains the gold option.

        Sizes run from ``min(min_size, n_options)`` to ``min(max_size,
        n_options)``, so the head also learns to rank a shortlist.  The cap
        keeps a large option set inside one sequence.  The gold is always
        present: a set without it has no correct answer to train against.
        """
        if not 0 <= gold_index < n_options:
            raise ValueError("gold_index is outside the option set")
        largest = min(n_options, max_size or n_options)
        size = rng.randint(min(min_size, largest), largest)
        others = [index for index in range(n_options) if index != gold_index]
        rng.shuffle(others)
        chosen = [gold_index] + others[: size - 1]
        rng.shuffle(chosen)
        return chosen
