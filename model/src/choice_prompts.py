"""Prompts for the cross-encoder choice head, fitted to the token budget.

The cross-encoder scores one prompt per option: ``state + instructions +
option``. ``_encode_text`` truncates from the right, and the option is the
suffix, so a state that fills the window silently drops the very text that
distinguishes the options — the model then scores a truncated option, or an
empty one, and the loss still looks normal. The tower head cannot fail this
way because it encodes each option on its own.

This module fits the prompts so the option always survives: it measures the
option (with the instructions) and shortens the *state* to leave room. The
token counter and decoder are injected so the arithmetic is testable without
torch or a tokenizer.
"""

from __future__ import annotations

from typing import Callable, Sequence

#: Tokens kept free on top of the measured suffix, for the space that joins
#: the state to the instructions and for tokenizer boundary effects (a string
#: concatenation can tokenise differently from its parts).
SAFETY_TOKENS = 2


def option_prompt(state: str, instructions: str, option_text: str) -> str:
    """The prompt scored for one option."""
    parts = [part for part in (state or "", instructions or "", option_text or "") if part.strip()]
    return " ".join(parts).strip()


def fit_option_prompts(
    state: str,
    instructions: str,
    option_texts: Sequence[str],
    max_length: int,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
) -> tuple[list[str], int]:
    """Build one prompt per option, shortening the state so options survive.

    Args:
        state: The item's state text, the only part that may be shortened.
        instructions: The question text; never shortened.
        option_texts: One text per option; never shortened.
        max_length: The encoder's token limit.
        encode: ``text -> token ids``.
        decode: ``token ids -> text``.

    Returns:
        ``(prompts, dropped_state_tokens)``. The second value is 0 when the
        state fitted whole, and otherwise how many of its leading tokens were
        dropped — callers warn once so a run cannot lose options silently.
    """
    suffix_ids = [
        encode(" ".join(part for part in ((instructions or "").strip(), (text or "").strip()) if part))
        for text in option_texts
    ]
    worst_suffix = max((len(ids) for ids in suffix_ids), default=0)
    budget = max_length - worst_suffix - SAFETY_TOKENS

    state_ids = encode(state or "")
    if budget <= 0:
        # Even the options alone do not fit; there is nothing sensible to keep.
        # Return the plain prompts and report every state token as dropped so
        # the caller warns rather than pretending the measurement is clean.
        return [option_prompt(state, instructions, text) for text in option_texts], len(state_ids)

    if len(state_ids) <= budget:
        return [option_prompt(state, instructions, text) for text in option_texts], 0

    # Keep the opening of the state: the instructions and the option carry the
    # question, and a document's first tokens usually set the frame.
    kept = decode(state_ids[:budget])
    prompts = [option_prompt(kept, instructions, text) for text in option_texts]
    return prompts, len(state_ids) - budget
