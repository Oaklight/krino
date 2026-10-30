"""Length alone: pad a state with neutral text and re-score the same question.

`research/13` shows accuracy falling about 30 points from the shortest to the
longest quartile of JevBench hard-tier states, for the trained head and the
backbone's native readout alike. That comparison is across items, so length is
confounded with whatever else makes those items hard. And the corpus audit in
the same note shows the conjunction is nearly absent from training: 25 of 69,287
choice items reach the tier's median length.

This module holds the part of the follow-up that can be tested without a model:
take an item, append neutral text until its state reaches a target length, and
leave everything else — the question, the options, the gold label — untouched.
If accuracy falls as padding grows, length is the variable. If it holds, the
tier's long states carry something the model cannot use that length alone does
not reproduce.

The filler is deliberately contentless and repeated: it is not a second
document, so the only thing it adds is tokens.
"""

from __future__ import annotations

CLOSING = (
    "The above notes were kept for reference and are not part of the question."
)

# Two sentences of ordinary prose, repeated to reach a target length. Kept
# dull on purpose: padding that read like another document could either help
# (extra context) or hurt (distraction), and neither is the variable here.
_FILLER = (
    "This paragraph is archive material kept alongside the record. "
    "It restates earlier notes and carries no new information about the case. "
)


def pad_state(state: str, target_chars: int, filler: str = _FILLER) -> str:
    """Append filler to ``state`` until it is at least ``target_chars`` long.

    Padding goes *after* the state rather than before it, because a causal
    backbone pools the last token and prefix padding would change what the
    pooled position sees; suffix padding leaves every original token where it
    was and only lengthens the sequence the pooled position has to read over.

    Args:
        state: The original state.
        target_chars: Length to reach, in characters. A state already at least
            this long is returned unchanged.
        filler: The text to repeat. Must be non-empty.

    Returns:
        The padded state, ending with :data:`CLOSING` so the question does not
        read as continuing into the filler.

    Raises:
        ValueError: If ``filler`` is empty.
    """
    if not filler:
        raise ValueError("filler must be non-empty")
    if len(state) >= target_chars:
        return state
    # The closing sentence is part of what the model reads, so it comes out of
    # the budget rather than being added on top of it.
    body_budget = max(target_chars - len(state) - len(CLOSING) - 1, 0)
    reps = -(-body_budget // len(filler))  # ceil
    body = filler * reps
    return f"{state} {body[:body_budget]} {CLOSING}" if body else f"{state} {CLOSING}"
