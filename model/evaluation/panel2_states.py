"""Panel-2 option spans and the state-cache contract (#340).

The set-head arm reads one hidden state per option from a single panel-2
forward, at the last token of that option's text.  This module holds the
torch-free part: where each option's text sits in the shipped prompt, and
which token is its last.  The extractor and its tests share one definition of
the span, so a state cannot be read from the wrong token without a test
failing.

The cache contract is also here: :func:`cache_key` names what the stored
states depend on, and a manifest records the same fields beside the arrays.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from model.evaluation.panel2 import PANEL2_TEMPLATE, render_panel2

# The 26-letter alphabet covers every corpus and JevBench item.  The committed
# development cell (#326) also holds 24 banking77 items with 80 options, and its
# panel-2 reference was read with this 96-symbol alphabet, where each code is
# one token; the first 26 entries are the same letters main ships.
EXTENDED_CODES = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!@#$%^&*()-_=+[]{}<>?|~"
    "\\/\"'`"
    ",.;:"
)


def prompt_codes(n_options: int) -> list[str]:
    """The codes for ``n_options``; identical to ``codes_for`` up to 26."""
    if not 2 <= n_options <= len(EXTENDED_CODES):
        raise ValueError(
            f"n_options must be in [2, {len(EXTENDED_CODES)}], got {n_options}"
        )
    return list(EXTENDED_CODES[:n_options])


def render_with_spans(
    state: str,
    instructions: str,
    texts: Sequence[str],
    codes: Sequence[str] | None = None,
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """The panel-2 prompt and the character span of each option text.

    The prompt is exactly :func:`render_panel2`'s output; the spans are checked
    against it, so a rendering change cannot silently move them.

    Args:
        state: The item's state text.
        instructions: The question text.
        texts: Option texts in display order.
        codes: One code per option; the shipped A..Z by default.

    Returns:
        ``(prompt, codes, spans)`` where ``spans[i]`` is the half-open
        ``(start, end)`` of option ``i``'s text inside ``prompt``.
    """
    codes = list(codes) if codes is not None else prompt_codes(len(texts))
    if len(codes) != len(texts):
        raise ValueError(f"{len(texts)} option texts but {len(codes)} codes")
    prompt = render_panel2(state, instructions, texts, codes)
    header = "State: " + state + "\nQuestion: " + instructions + "\nOptions:\n"
    if not prompt.startswith(header):
        raise ValueError("prompt does not start with the expected header")
    spans: list[tuple[int, int]] = []
    cursor = len(header)
    for code, text in zip(codes, texts):
        line = f"{code}. {text}"
        start = cursor + len(code) + 2
        end = start + len(text)
        if prompt[start:end] != text:
            raise ValueError(
                f"option {code}: span {start}:{end} does not hold {text!r} "
                f"but {prompt[start:end]!r}"
            )
        spans.append((start, end))
        cursor += len(line) + 1  # the newline after the option line
    return prompt, codes, spans


def last_token_index(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> int:
    """The last token that overlaps ``span``.

    Args:
        offsets: ``(start, end)`` per token from the tokenizer's offset map.
        span: The option text's character span.

    Returns:
        The token index, or ``-1`` when no token overlaps the span.
    """
    start, end = span
    for index in range(len(offsets) - 1, -1, -1):
        token_start, token_end = offsets[index]
        if token_start < end and token_end > start:
            return index
    return -1


def option_token_indices(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> list[int]:
    """The last-token index of every option; every index must exist."""
    indices = [last_token_index(offsets, span) for span in spans]
    if any(index < 0 for index in indices):
        raise ValueError(f"an option text has no token: indices {indices}")
    return indices


def data_digest(rows: Sequence[dict]) -> str:
    """Hash of the identities, option texts and gold indices the cache holds."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            repr([row["id"], list(row["texts"]), int(row["gold"])]).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def cache_key(
    model_revision: str,
    tokenizer_digest: str,
    order: str,
) -> str:
    """The string a cache manifest must carry to be reusable.

    Model revision and tokenizer digest both matter: the tokenizer decides
    which token ends an option, and the model revision decides the state.  The
    prompt template is hashed in as well, because a template change moves every
    span.
    """
    template = hashlib.sha256(PANEL2_TEMPLATE.encode("utf-8")).hexdigest()
    payload = "|".join([model_revision, tokenizer_digest, template, order])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_for(
    row: dict, order: str
) -> tuple[str, list[str], list[str], list[tuple[int, int]]]:
    """The prompt, its codes, the texts in display order and each option's span."""
    codes = prompt_codes(len(row["texts"]))
    texts = list(row["texts"])
    if order == "reversed":
        texts = list(reversed(texts))
    prompt, codes, spans = render_with_spans(
        row["state"], row["instructions"], texts, codes
    )
    if prompt != render_panel2(row["state"], row["instructions"], texts, codes):
        raise AssertionError("span renderer disagrees with render_panel2")
    return prompt, codes, texts, spans


def build_batches(
    rows: Sequence[dict],
    tokenizer,
    batch_size: int,
    max_tokens: int,
    max_prompt_tokens: int,
) -> tuple[list[list[dict]], list[dict]]:
    """Fixed batches: grouped by split and K, longest prompts first.

    An item whose prompt is longer than ``max_prompt_tokens`` is skipped and
    returned in the second list: a long sequence makes attention cost dominate
    and its last-token state is not worth the wait.  A batch also stops at
    ``max_tokens`` summed prompt tokens, so one group of long options cannot
    ask for an activation set the GPU cannot hold.  A single item over the
    budget still forms its own batch.

    Args:
        rows: Cache rows with ``id``, ``split``, ``gold``, ``texts``,
            ``state``, ``instructions`` and ``order``.
        tokenizer: Callable whose ``__call__(text, add_special_tokens=False)``
            returns a mapping with ``input_ids``.
        batch_size: Maximum items per batch.
        max_tokens: Maximum summed prompt tokens per batch.
        max_prompt_tokens: Items longer than this are skipped.

    Returns:
        ``(batches, skipped)``; ``skipped`` carries ``id``, ``split`` and
        ``tokens`` per excluded item.
    """
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    skipped: list[dict] = []
    for index, row in enumerate(rows):
        prompt, codes, texts, spans = prompt_for(row, row["order"])
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length > max_prompt_tokens:
            skipped.append({"id": row["id"], "split": row["split"], "tokens": length})
            continue
        groups[(row["split"], len(texts))].append(
            {
                "index": index,
                "id": row["id"],
                "gold": row["gold"],
                "order": row["order"],
                "prompt": prompt,
                "spans": spans,
                "length": length,
            }
        )
    batches: list[list[dict]] = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda item: (-item["length"], item["index"]))
        batch: list[dict] = []
        tokens = 0
        for item in items:
            if batch and (
                len(batch) >= batch_size or tokens + item["length"] > max_tokens
            ):
                batches.append(batch)
                batch, tokens = [], 0
            batch.append(item)
            tokens += item["length"]
        if batch:
            batches.append(batch)
    return batches, skipped
