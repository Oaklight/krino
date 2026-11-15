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

import copy
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

    #: Suffix rows per forward.  Attention memory grows with the suffix rows
    #: times the padded prefix length, so a step's rows are scored in blocks of
    #: this size; a row with more options than this is split, and its margins
    #: are concatenated in option order.
    SUFFIX_ROWS_PER_FORWARD = 8

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

    def _ids_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize many strings in one call; the fast tokenizer parallelises.

        The trainer tokenises every prompt of every step, and one call per
        string left the CPU as the step's largest single cost.  A list call
        runs the same Rust tokenizer over the batch.
        """
        return self.tokenizer(
            list(texts), add_special_tokens=False
        ).input_ids

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

    def lengths(self, rows: Sequence[Sequence[JudgementPrompt]]) -> list[int]:
        """Longest prompt of each row in tokens, tokenised in one batch.

        The chunk budget needs the real token length: a character estimate is
        wrong by more than 2x on non-Latin text, and the difference is the
        difference between a 20 GiB chunk and a 90 GiB one.
        """
        flat = [prompt.text for prompts in rows for prompt in prompts]
        ids = self._ids_batch(flat)
        out: list[int] = []
        cursor = 0
        for prompts in rows:
            block = ids[cursor : cursor + len(prompts)]
            cursor += len(prompts)
            out.append(max((len(part) for part in block), default=0))
        return out

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
        if any(prompt.prefix != prompts[0].prefix for prompt in prompts):
            raise ValueError(f"row {prompts[0].row_id} prefixes are not identical")
        encoded = self._ids_batch(
            [prompts[0].prefix]
            + [prompt.suffix for prompt in prompts]
            + [prompts[0].text]
        )
        prefix_ids = encoded[0]
        suffix_ids = encoded[1:-1]
        if encoded[-1] != prefix_ids + suffix_ids[0]:
            raise ValueError(
                f"prompt {prompts[0].row_id}#0 does not tokenise as prefix + "
                "suffix; the prefix-cache path would train a different token "
                "sequence from inference"
            )
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

    def step_margins(
        self, rows: Sequence[Sequence[JudgementPrompt]]
    ) -> list[torch.Tensor]:
        """Score every row of one optimizer step with two batched forwards.

        The K prompts of a row share one prefix, and the rows of a step are
        independent, so the prefix forward runs once for the whole step (a
        batch of row prefixes) and the suffix forward runs once for every
        option of every row.  One backward can then cover the step.  The
        per-row :meth:`row_margins` path launches two small forwards and one
        backward per row, and the card sits idle between those launches.

        Returns one ``[K]`` margin tensor per input row, in input order.  A row
        with an empty prefix (a criteria-less noul) has no cache to batch and
        is scored by :meth:`row_margins`; mixing it into the prefix batch would
        leave pad queries with no key at all, whose NaN K/V would poison the
        suffix attention.
        """
        if not rows or any(not prompts for prompts in rows):
            raise ValueError("step_margins needs non-empty rows")
        margins: list[torch.Tensor | None] = [None] * len(rows)
        batched: list[int] = []
        for index, prompts in enumerate(rows):
            prefix = prompts[0].prefix
            if any(prompt.prefix != prefix for prompt in prompts):
                raise ValueError(
                    f"row {prompts[0].row_id} has option prompts with different "
                    "prefixes; the shared prefix pass would be wrong"
                )
            if not prefix:
                margins[index] = self.row_margins(prompts)
            else:
                batched.append(index)
        if batched:
            for index, margin in zip(
                batched, self._batched_rows_margins([rows[i] for i in batched])
            ):
                margins[index] = margin
        if any(margin is None for margin in margins):
            raise RuntimeError("step_margins left a row unscored")
        return margins

    def _batched_rows_margins(
        self, rows: Sequence[Sequence[JudgementPrompt]]
    ) -> list[torch.Tensor]:
        for prompts in rows:
            if any(prompt.prefix != prompts[0].prefix for prompt in prompts):
                raise ValueError(f"row {prompts[0].row_id} prefixes are not identical")
        flat = [prompt for prompts in rows for prompt in prompts]
        encoded = self._ids_batch(
            [prompts[0].prefix for prompts in rows]
            + [prompt.suffix for prompt in flat]
            + [prompts[0].text for prompts in rows]
        )
        rows_count = len(rows)
        prefixes = encoded[:rows_count]
        flat_suffixes = encoded[rows_count : rows_count + len(flat)]
        texts = encoded[rows_count + len(flat) :]
        suffixes: list[list[list[int]]] = []
        cursor = 0
        for prompts, prefix_ids, text_ids in zip(rows, prefixes, texts):
            row_suffixes = flat_suffixes[cursor : cursor + len(prompts)]
            cursor += len(prompts)
            if text_ids != prefix_ids + row_suffixes[0]:
                raise ValueError(
                    f"prompt {prompts[0].row_id}#0 does not tokenise as "
                    "prefix + suffix; the prefix-cache path would train a "
                    "different token sequence from inference"
                )
            suffixes.append(row_suffixes)

        # Group by prefix bucket.  One batch per bucket keeps every row's
        # suffix attention over a prefix of its own bucket's length; a single
        # batch for the whole step would make short rows attend over the
        # longest row's bucket, which costs more than the saved launches.
        groups: dict[int, list[int]] = {}
        for index, row in enumerate(suffixes):
            if not self.bucket_shapes:
                groups.setdefault(len(prefixes[index]), []).append(index)
                continue
            width = self._width_bucket(max(len(ids) for ids in row))
            prefix_bucket = bucket_length(
                len(prefixes[index]), PREFIX_BUCKETS, limit=self.max_length - width
            )
            if prefix_bucket + width > self.max_length:
                prefix_bucket = len(prefixes[index])
            groups.setdefault(prefix_bucket, []).append(index)

        margins: list[torch.Tensor | None] = [None] * rows_count
        for prefix_bucket, indices in groups.items():
            if len(indices) == 1:
                margins[indices[0]] = self.row_margins(rows[indices[0]])
                continue
            for index, margin in zip(
                indices,
                self._prefix_group_margins(
                    [prefixes[i] for i in indices],
                    [suffixes[i] for i in indices],
                    prefix_bucket,
                ),
            ):
                margins[index] = margin
        if any(margin is None for margin in margins):
            raise RuntimeError("step_margins left a row unscored")
        return margins

    def _prefix_group_margins(
        self,
        prefixes: list[list[int]],
        suffixes: list[list[list[int]]],
        prefix_bucket: int,
    ) -> list[torch.Tensor]:
        """One prefix forward and one suffix forward for rows of one bucket."""
        if not self.bucket_shapes:
            width = max(len(ids) for row in suffixes for ids in row)
            prefix_bucket = max(len(ids) for ids in prefixes)
        else:
            width = self._width_bucket(max(len(ids) for row in suffixes for ids in row))
        if prefix_bucket + width > self.max_length:
            prefix_bucket = max(len(ids) for ids in prefixes)
        if any(
            len(prefix) + len(suffix) > self.max_length
            for prefix, row in zip(prefixes, suffixes)
            for suffix in row
        ):
            raise ValueError(
                f"a row of this step needs more than {self.max_length} tokens"
            )
        cache = self._prefix_batch(prefixes, prefix_bucket)
        pairs = [
            (index, suffix)
            for index, row in enumerate(suffixes)
            for suffix in row
        ]
        margins: list[list[torch.Tensor]] = [[] for _ in suffixes]
        for start in range(0, len(pairs), self.SUFFIX_ROWS_PER_FORWARD):
            block = pairs[start : start + self.SUFFIX_ROWS_PER_FORWARD]
            real = len(block)
            # Pad to a fixed shape so the kernel sees one width per bucket; the
            # repeated trailing pair is never read.
            block = block + [block[-1]] * (self.SUFFIX_ROWS_PER_FORWARD - real)
            combined = self._broadcast_cache(cache, [index for index, _ in block])
            ids, mask, positions = self._suffix_block_inputs(
                prefixes, block, prefix_bucket, width
            )
            cache_position = torch.arange(
                prefix_bucket, prefix_bucket + width, device=self.device
            )
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=combined,
                cache_position=cache_position,
                use_cache=True,
            )
            logits = outputs.logits
            for offset in range(real):
                index, suffix = block[offset]
                margins[index].append(
                    margin_from_logits(
                        logits[offset, len(suffix) - 1], self.yes_id, self.no_id
                    )
                )
        return [torch.stack(parts) for parts in margins]

    def _suffix_block_inputs(
        self,
        prefixes: list[list[int]],
        block: list[tuple[int, list[int]]],
        prefix_bucket: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = self._pad_id()
        rows = len(block)
        ids = torch.full((rows, width), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros(
            (rows, prefix_bucket + width), dtype=torch.long, device=self.device
        )
        positions = torch.zeros((rows, width), dtype=torch.long, device=self.device)
        for offset, (index, suffix) in enumerate(block):
            prefix_len = len(prefixes[index])
            length = len(suffix)
            ids[offset, :length] = torch.tensor(suffix, device=self.device)
            mask[offset, :prefix_len] = 1
            mask[offset, prefix_bucket : prefix_bucket + length] = 1
            positions[offset, :length] = torch.arange(
                prefix_len, prefix_len + length, device=self.device
            )
        return ids, mask, positions

    def _prefix_batch(self, prefixes: list[list[int]], bucket: int):
        """One prefix forward for a batch of row prefixes, right-padded."""
        rows = len(prefixes)
        pad_id = self._pad_id()
        ids = torch.full((rows, bucket), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((rows, bucket), dtype=torch.long, device=self.device)
        positions = torch.zeros((rows, bucket), dtype=torch.long, device=self.device)
        for index, prefix in enumerate(prefixes):
            length = len(prefix)
            ids[index, :length] = torch.tensor(prefix, device=self.device)
            mask[index, :length] = 1
            positions[index, :length] = torch.arange(length, device=self.device)
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

    def _broadcast_cache(self, cache, owners: list[int]):
        """A cache whose row ``i`` is row ``owners[i]`` of ``cache``."""
        from transformers.cache_utils import DynamicCache

        combined = DynamicCache()
        index = torch.tensor(owners, device=self.device)
        for layer in cache.layers:
            fresh = copy.copy(layer)
            fresh.keys = layer.keys.index_select(0, index)
            fresh.values = layer.values.index_select(0, index)
            fresh.is_initialized = True
            combined.layers.append(fresh)
        return combined

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
