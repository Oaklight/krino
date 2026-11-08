"""Label semantics for canonical corpus rows.

A choice label is either a criteria key (``option``) or, for a constructed
abstention row, the withheld gold (``withheld_gold``).  A score label is either
an integer level (``level``) or a float expected level between two levels
(``fractional``).  Noul rows have no label kind.

The builder writes ``label_kind`` on every choice and score row.  An absent
field is not allowed to mean ``option`` or ``level``: a consumer must not have to
infer the meaning of a label it cannot interpret.
"""

from __future__ import annotations

from typing import Any, Mapping

OPTION = "option"
WITHHELD_GOLD = "withheld_gold"
LEVEL = "level"
FRACTIONAL = "fractional"

CHOICE_KINDS = (OPTION, WITHHELD_GOLD)
SCORE_KINDS = (LEVEL, FRACTIONAL)

__all__ = [
    "CHOICE_KINDS",
    "FRACTIONAL",
    "LEVEL",
    "LabelError",
    "OPTION",
    "SCORE_KINDS",
    "WITHHELD_GOLD",
    "annotate",
    "label_kind",
]


class LabelError(ValueError):
    """A row whose label cannot be interpreted under the corpus contract."""


def label_kind(item: Mapping[str, Any]) -> str | None:
    """Return the label kind, or raise :class:`LabelError` with the row address.

    Choice: the label must be a criteria key, or the row must be an abstention
    row (``answerable is False``) whose gold was removed on purpose.  Score: the
    label must be numeric and inside ``0..len(criteria)-1``; an integer is a
    ``level``, anything else is a ``fractional`` expected level.  Noul: no kind.
    """
    question = item.get("question") or {}
    q_type = question.get("type")
    item_id = item.get("id")

    if q_type == "choice":
        criteria = question.get("criteria") or {}
        label = item.get("label")
        if label in criteria:
            return OPTION
        if item.get("answerable") is False:
            return WITHHELD_GOLD
        raise LabelError(
            f"{item_id}: choice label {label!r} is not a criteria key "
            f"(keys: {sorted(criteria)}) and answerable is not False"
        )

    if q_type == "score":
        criteria = question.get("criteria") or []
        label = item.get("label")
        try:
            value = float(label)
        except (TypeError, ValueError) as exc:
            raise LabelError(
                f"{item_id}: score label {label!r} is not numeric"
            ) from exc
        upper = len(criteria) - 1
        if not 0.0 <= value <= float(upper):
            raise LabelError(
                f"{item_id}: score label {value} is outside 0..{upper}"
            )
        return LEVEL if abs(value - round(value)) < 1e-9 else FRACTIONAL

    return None


def annotate(item: dict[str, Any]) -> dict[str, Any]:
    """Write ``label_kind`` on a choice or score row and return the row."""
    question = item.get("question") or {}
    if question.get("type") in ("choice", "score"):
        item["label_kind"] = label_kind(item)
    return item
