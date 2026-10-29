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


def _as_text(value: object) -> str:
    """Whatever the caller holds, as prompt text.

    Items carry their state as whatever the loader left there: a string for most
    sources, a structured object for 35 of JevBench's 111 hard items. Rendering
    those is the loader's job (``data.pipeline.render_jevbench_state``, #180);
    this only guarantees the prompt builder does not crash on one and lose the
    option. A dict that reaches ``str`` is Python's ``repr``, which is a poor
    prompt but still contains the state, and it is what the tower path has
    always built from an f-string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def option_prompt(state: str, instructions: str, option_text: str) -> str:
    """The prompt scored for one option."""
    parts = [
        part
        for part in (
            _as_text(state),
            _as_text(instructions),
            _as_text(option_text),
        )
        if part.strip()
    ]
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

    Non-string inputs are coerced (see :func:`_as_text`), so a structured state
    from a loader that did not render it produces a prompt rather than a
    tokenizer ``TypeError``.

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
    state = _as_text(state)
    instructions = _as_text(instructions)
    option_texts = [_as_text(text) for text in option_texts]
    suffix_ids = [
        encode(" ".join(part for part in (instructions.strip(), text.strip()) if part))
        for text in option_texts
    ]
    worst_suffix = max((len(ids) for ids in suffix_ids), default=0)
    budget = max_length - worst_suffix - SAFETY_TOKENS

    state_ids = encode(state)
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
