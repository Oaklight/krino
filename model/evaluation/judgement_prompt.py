"""Prompt and option construction for the per-option judgement readout.

The continuation readout asks which text is most likely to follow ``Answer:``.
The judgement readout states the option and asks a direct yes/no question about
it, which is the form Qwen3-Reranker is trained on and the form a base model has
seen far more often than a completion cue.

This module holds the parts that decide *what is asked*, with no torch import, so
the script that scores and the test that checks the construction share one
definition rather than two that can drift.
"""

from __future__ import annotations

from typing import Any

JUDGEMENT_TEMPLATE = (
    "State: {state}\n"
    "Question: {instructions}\n"
    "Option: {option}\n"
    "Does this option correctly answer the question? Answer:"
)

YES = " yes"
NO = " no"


def candidates(item: Any) -> tuple[list[str], int] | None:
    """Return the option texts presented to the judgement, and the gold index.

    The text placed after ``Option:`` is the text the other readouts score: the
    noul answer wording from :func:`noul_option_texts`, so the 74 noul items keep
    their own phrasing; the choice description; and a named rubric level for
    score items. ``None`` means the item's label is not one of its options.
    """
    question = item.question
    q_type = question.get("type", "noul")

    if q_type == "noul":
        from model.evaluation.prompt_options import noul_option_texts

        texts = [text.strip() for text in noul_option_texts(question.get("criteria"))]
        positive = (
            item.label
            if isinstance(item.label, bool)
            else str(item.label).lower() == "true"
        )
        return texts, 0 if positive else 1

    if q_type == "choice":
        criteria = question["criteria"]
        if item.label not in criteria:
            return None
        texts = [str(desc or key) for key, desc in criteria.items()]
        return texts, list(criteria.keys()).index(item.label)

    if q_type == "score":
        criteria = question["criteria"]
        texts = [f"Level {index}: {desc}" for index, desc in enumerate(criteria)]
        gold = max(0, min(len(criteria) - 1, int(round(float(item.label)))))
        return texts, gold

    return None


def render(state: str, instructions: str, option: str) -> str:
    """One option's judgement prompt."""
    return JUDGEMENT_TEMPLATE.format(
        state=state, instructions=instructions, option=option
    )
