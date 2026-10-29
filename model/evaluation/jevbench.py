"""JevBench scoring: the accuracy rule and the per-tier aggregation.

JevBench is scored in two places. ``model/scripts/eval_jevbench.py`` calls
``DecisionModel.predict`` on a published HuggingFace checkpoint, which rounds
probabilities to two decimals; ``model/scripts/eval_calibration.py`` scores a
local training checkpoint from logits, which is what the goal's numbers have to
come from. The two paths must agree on what counts as correct and on which tier
an item belongs to, so both rules live here rather than in either script.
"""

from __future__ import annotations

from typing import Any, Sequence

from model.evaluation.calibration import softmax
from model.evaluation.order_sensitivity import tie_stable_argmax

TIERS = ("easy", "hard", "original")


def tier_from_id(item_id: str) -> str:
    """Return the tier an item belongs to, read from its id prefix.

    JevBench ids are ``<tier>-<n>``; anything unexpected is reported under its
    own prefix rather than dropped, so a renamed tier shows up as a row instead
    of silently shrinking the denominator.
    """
    return str(item_id).split("-")[0]


def record_correct(
    record: dict[str, Any],
    temperature: float = 1.0,
    prediction: int | None = None,
) -> bool:
    """Whether one prediction record is correct.

    Ties at the exact maximum are broken by the option's original index, via
    :func:`tie_stable_argmax`, so reordering the options cannot change whether
    an item counts as correct.

    Args:
        record: A record with ``logits``, ``label_order`` and ``label_idx``.
        temperature: Divisor applied to the logits before the softmax.
        prediction: The option index already chosen for this record. Supplied by
            callers that score in bulk, so the rule is shared without paying for
            a second softmax.
    """
    if prediction is None:
        probabilities = softmax([value / temperature for value in record["logits"]])
        prediction = tie_stable_argmax(probabilities, record["label_order"])
    return prediction == record["label_idx"]


def summarize(
    records: Sequence[dict[str, Any]],
    temperature: float = 1.0,
    failed: int = 0,
) -> dict[str, Any]:
    """Aggregate prediction records by tier, by type, and by both.

    Args:
        records: Records from ``_predict_original``; each carries ``_item``.
        temperature: Divisor applied to the logits before the softmax.
        failed: Items the caller could not score at all, reported rather than
            dropped so the denominator is visible.

    Returns:
        ``{"overall", "by_tier", "by_type", "by_tier_type"}`` where each leaf is
        ``{"accuracy", "correct", "total"}``. Tiers and types with no items are
        absent, so a missing row means the tier was not present in the data
        rather than that it scored zero.
    """
    overall = {"correct": 0, "total": 0, "failed": failed}
    by_tier: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}
    by_tier_type: dict[str, dict[str, dict[str, int]]] = {}

    for record in records:
        item = record["_item"]
        tier = tier_from_id(item.id)
        q_type = record.get("type", item.question.get("type", "noul"))
        correct = record_correct(record, temperature)

        overall["total"] += 1
        if correct:
            overall["correct"] += 1
        for bucket, key in ((by_tier, tier), (by_type, q_type)):
            entry = bucket.setdefault(key, {"correct": 0, "total": 0})
            entry["total"] += 1
            if correct:
                entry["correct"] += 1
        nested = by_tier_type.setdefault(tier, {}).setdefault(
            q_type, {"correct": 0, "total": 0}
        )
        nested["total"] += 1
        if correct:
            nested["correct"] += 1

    def _rate(entry: dict[str, int]) -> dict[str, Any]:
        total = entry["total"]
        return {
            "accuracy": entry["correct"] / total if total else 0.0,
            "correct": entry["correct"],
            "total": total,
        }

    return {
        "overall": {**_rate(overall), "failed": overall["failed"]},
        "by_tier": {tier: _rate(entry) for tier, entry in by_tier.items()},
        "by_type": {q_type: _rate(entry) for q_type, entry in by_type.items()},
        "by_tier_type": {
            tier: {q_type: _rate(entry) for q_type, entry in types.items()}
            for tier, types in by_tier_type.items()
        },
    }


def prediction_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The option each item was predicted to, not only whether it was right.

    Two checkpoints can score the same accuracy on the same items for two very
    different reasons -- a shared capability, or a shared bias toward one option
    key. Accuracy alone cannot tell them apart; the predicted option can, so it
    is written next to the summary.

    Args:
        records: Records from ``_predict_original``.

    Returns:
        One row per record with the item id, its type, the gold and predicted
        option labels, and whether they match. Labels are index-based when a
        record carries no ``labels`` list.
    """
    rows = []
    for record in records:
        probabilities = softmax(record["logits"])
        prediction = tie_stable_argmax(probabilities, record["label_order"])
        labels = list(record.get("labels") or [])
        gold = record["label_idx"]
        rows.append(
            {
                "id": getattr(record.get("_item"), "id", None),
                "type": record.get("type"),
                "n_options": len(labels),
                "gold": labels[gold] if gold < len(labels) else gold,
                "predicted": labels[prediction]
                if prediction < len(labels)
                else prediction,
                "correct": prediction == gold,
            }
        )
    return rows
