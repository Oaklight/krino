"""Batched prefix/suffix readout for the judgement objective (#299).

One row produces K prompts, one per option.  Every prompt of a row shares the
byte-identical prefix (``State`` + ``Question``), so the trainer encodes that
prefix once with a KV cache and then scores the K option suffixes as one
batched forward.  That is the #299 throughput item: 243,366 rows at about 3.7
options each is roughly 900k option forwards per epoch, and re-encoding the
prefix per option would make an epoch unaffordable.

The readout emits the raw ``logit(" yes") - logit(" no")`` at each prompt's final
position.  It does not normalise, scale, or threshold; :class:`ScaledMargin` and
the row-normalised loss in :mod:`model.src.judgement_objective` do that.

The prompt text comes from :mod:`model.evaluation.judgement_prompt`, and the
token ids are the ones the scoring path would see: the readout asserts that
``tokenize(prefix) + tokenize(suffix)`` equals ``tokenize(text)`` for every
prompt and raises instead of silently training a different prompt.  That
assertion is the drift guard between training and inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model.evaluation.judgement_prompt import NO, YES
from model.src.judgement_objective import JudgementPrompt, margin_from_logits
from model.src.shape_buckets import (
    PREFIX_BUCKETS,
    SUFFIX_BUCKETS,
    bucket_length,
    bucket_rows,
)


class JudgementReadout:
    """Score the K option prompts of one row with one shared prefix pass.

    Args:
        model: Causal LM (optionally LoRA-adapted) whose next-token logits
            carry the yes/no answer.
        tokenizer: Its tokenizer.
        max_length: Longest ``prefix + suffix`` accepted.  A longer row raises;
            the trainer drops it and reports the count so truncation never
            silently changes a prompt.
    """

    def __init__(
        self, model, tokenizer, max_length: int = 4096, bucket_shapes: bool = True
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bucket_shapes = bucket_shapes
        self.device = next(model.parameters()).device
        self.yes_id = self._single_token(YES)
        self.no_id = self._single_token(NO)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _single_token(self, text: str) -> int:
        ids = self._ids(text)
        if len(ids) != 1:
            raise ValueError(
                f"the answer token {text!r} is {len(ids)} tokens, not one; "
                "the judgement margin needs a single yes/no token"
            )
        return ids[0]

    def prompt_ids(self, prompt: JudgementPrompt) -> tuple[list[int], list[int]]:
        """``(prefix_ids, suffix_ids)`` whose concatenation is ``tokenize(text)``.

        Raises:
            ValueError: The prefix and suffix do not tokenise independently to
                the full prompt, which would make training score a different
                token sequence from the readout at inference.
        """
        prefix_ids = self._ids(prompt.prefix)
        suffix_ids = self._ids(prompt.suffix)
        text_ids = self._ids(prompt.text)
        if text_ids != prefix_ids + suffix_ids:
            raise ValueError(
                f"prompt {prompt.row_id}#{prompt.option_index} does not tokenise "
                "as prefix + suffix; the prefix-cache path would train a "
                "different token sequence from inference"
            )
        return prefix_ids, suffix_ids

    def fits(self, prompts: Sequence[JudgementPrompt]) -> bool:
        """Whether every prompt of the row fits ``max_length``.

        Length only.  A prefix/suffix tokenisation mismatch is a different
        error and is raised by :meth:`prompt_ids`, never silently dropped.
        """
        if not prompts:
            return False
        return max(len(self._ids(prompt.text)) for prompt in prompts) <= self.max_length

    def row_margins(self, prompts: Sequence[JudgementPrompt]) -> torch.Tensor:
        """Raw yes-minus-no margins, ``[K]``, one per prompt of the row."""
        if not prompts:
            raise ValueError("row_margins needs at least one prompt")
        prefix = prompts[0].prefix
        if any(prompt.prefix != prefix for prompt in prompts):
            raise ValueError(
                f"row {prompts[0].row_id} has option prompts with different "
                "prefixes; the shared prefix pass would be wrong"
            )
        splits = [self.prompt_ids(prompt) for prompt in prompts]
        prefix_ids = splits[0][0]
        suffix_ids = [suffix for _, suffix in splits]
        if any(ids != prefix_ids for ids, _ in splits):
            raise ValueError(f"row {prompts[0].row_id} prefixes are not identical")
        longest = max(len(ids) for ids in suffix_ids)
        if len(prefix_ids) + longest > self.max_length:
            raise ValueError(
                f"row {prompts[0].row_id} needs {len(prefix_ids) + longest} "
                f"tokens, over max_length={self.max_length}"
            )

        # Fixed shapes: a new (prefix, suffix, row-count) shape costs a 1.5-3.4 s
        # cold kernel load on this class of accelerator, which dominates a step
        # whose real compute is tens of milliseconds.  Pad every row into the
        # bucket grid so each shape is paid once.  Padding is masked out, so
        # the real tokens see the same computation as an unpadded sequence.
        count = len(suffix_ids)
        longest = max(len(ids) for ids in suffix_ids)
        if self.bucket_shapes:
            width = self._width_bucket(longest)
            # An empty prefix has no cache, so it must reserve no cache slots;
            # a non-zero bucket here misaligns the mask and cache_position.
            prefix_bucket = (
                bucket_length(
                    len(prefix_ids), PREFIX_BUCKETS, limit=self.max_length - width
                )
                if prefix_ids
                else 0
            )
            if prefix_bucket + width > self.max_length:
                # No bucket fits under max_length; fall back to exact shapes.
                prefix_bucket, width = len(prefix_ids), longest
            rows = bucket_rows(count)
        else:
            width = longest
            prefix_bucket = len(prefix_ids)
            rows = count
        cache = self._prefix_cache(prefix_ids, prefix_bucket) if prefix_ids else None
        if cache is not None:
            # One prefix graph, K suffix rows.  ``batch_repeat_interleave`` is a
            # differentiable repeat, so the prefix gradient is the sum over
            # options.
            cache.batch_repeat_interleave(rows)
        suffix, mask, positions = self._batch_inputs(
            len(prefix_ids), prefix_bucket, suffix_ids, width, rows
        )
        cache_position = torch.arange(
            prefix_bucket, prefix_bucket + width, device=self.device
        )
        outputs = self.model(
            input_ids=suffix,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = outputs.logits
        return torch.stack(
            [
                margin_from_logits(logits[index, len(ids) - 1], self.yes_id, self.no_id)
                for index, ids in enumerate(suffix_ids)
            ]
        )

    @staticmethod
    def _width_bucket(width: int) -> int:
        """Suffix width bucket; short option suffixes use the finer grid."""
        small = bucket_length(width, SUFFIX_BUCKETS)
        if small <= SUFFIX_BUCKETS[-1]:
            return small
        return bucket_length(width, PREFIX_BUCKETS)

    def _pad_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id or 0 if pad_id is None else pad_id

    def _prefix_cache(self, prefix_ids: list[int], bucket: int):
        # Right-pad: real tokens first.  A left-padded position 0 would be a
        # fully masked query (it can only see itself, which is masked), and an
        # all-masked query's softmax is NaN; the NaN lands in the cached K/V and
        # poisons every suffix through 0 * NaN.  With right padding the pad
        # queries see the real prefix, so they stay finite and their cache slots
        # are masked out of the suffix attention.
        pad = max(0, bucket - len(prefix_ids))
        ids = torch.tensor(
            [prefix_ids + [self._pad_id()] * pad],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros((1, bucket), dtype=torch.long, device=self.device)
        mask[0, : len(prefix_ids)] = 1
        positions = torch.zeros((1, bucket), dtype=torch.long, device=self.device)
        positions[0, : len(prefix_ids)] = torch.arange(
            len(prefix_ids), device=self.device
        )
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("the backbone returned no KV cache")
        return cache

    def _batch_inputs(
        self,
        prefix_len: int,
        prefix_bucket: int,
        suffix_ids: list[list[int]],
        width: int,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = self._pad_id()
        ids = torch.full((rows, width), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros(
            (rows, prefix_bucket + width), dtype=torch.long, device=self.device
        )
        positions = torch.zeros((rows, width), dtype=torch.long, device=self.device)
        # The real prefix sits at the start of the cache and is visible to every
        # row, including the padding rows.
        mask[:, :prefix_len] = 1
        for index in range(rows):
            suffix = suffix_ids[index] if index < len(suffix_ids) else suffix_ids[0]
            length = len(suffix)
            ids[index, :length] = torch.tensor(suffix, device=self.device)
            mask[index, prefix_bucket : prefix_bucket + length] = 1
            # RoPE positions are gap-free from the real prefix end, while
            # cache_position (below) indexes the padded cache.
            positions[index, :length] = torch.arange(
                prefix_len, prefix_len + length, device=self.device
            )
        return ids, mask, positions


def subsample_prompts(
    prompts: Sequence[JudgementPrompt], max_options: int, rng
) -> list[JudgementPrompt]:
    """Cap a row's options at ``max_options``, always keeping a positive.

    #299 prefers negative subsampling over dropping a many-option source: the
    77-way rows stay in training, but cannot dominate the prefix cache's memory
    or the row's cost.  The positive is kept so the row still has a target.
    """
    if len(prompts) <= max_options:
        return list(prompts)
    positives = [prompt for prompt in prompts if prompt.label == 1]
    negatives = [prompt for prompt in prompts if prompt.label != 1]
    keep_positive = positives[:1]
    rng.shuffle(negatives)
    chosen = keep_positive + negatives[: max_options - len(keep_positive)]
    keep = {prompt.option_index for prompt in chosen}
    return [prompt for prompt in prompts if prompt.option_index in keep]
