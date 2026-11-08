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

# A noul item without criteria has no option text worth judging: the question is
# the proposition, so the readout reads the answer at ``Answer:`` directly.
POLARITY_TEMPLATE = "State: {state}\nQuestion: {instructions}\nAnswer:"

YES = " yes"
NO = " no"


def has_criteria(criteria: Any) -> bool:
    """Whether a noul item carries its own answer wording."""
    return isinstance(criteria, dict) and bool(
        criteria.get("true") or criteria.get("false")
    )


def polarity_gold(item: Any) -> bool | None:
    """The gold answer for a criteria-less noul item, or ``None`` if it is not one.

    JevBench's 74 noul items all carry criteria, so this returns ``None`` for
    every acceptance-set item and the branch cannot be tuned against it.
    """
    question = item.question
    if question.get("type", "noul") != "noul":
        return None
    if has_criteria(question.get("criteria")):
        return None
    label = item.label
    if isinstance(label, bool):
        return label
    text = str(label).strip().lower()
    if text in ("true", "yes", "1", "y"):
        return True
    if text in ("false", "no", "0", "n"):
        return False
    return None


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

        criteria = question.get("criteria")
        if not has_criteria(criteria):
            # No option text to judge; the caller reads polarity at ``Answer:``.
            return None
        texts = [text.strip() for text in noul_option_texts(criteria)]
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


def render_polarity(state: str, instructions: str) -> str:
    """The one-proposition prompt: the question is answered at ``Answer:``."""
    return POLARITY_TEMPLATE.format(state=state, instructions=instructions)
