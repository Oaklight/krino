"""Cascade decision rule: panel 2 first, a judgement re-read of the close calls.

Arm F (#342).  The first pass is the shipped panel-2 readout: one forward, K
code logits, a prediction.  The second pass is the per-option judgement readout
on the top two candidates, two forwards.  The cascade takes the first-pass
answer unless the top-2 margin is below a threshold.

Everything that decides an answer is here, with no torch import, so the sweep,
the evaluation and the tests share one definition:

* the margin is a difference of **probabilities** obtained by softmax over the
  K code logits, ``p1 - p2``, never a difference of logits;
* the top two are ordered by ``(-probability, natural index)``;
* an exact tie in the second pass resolves to the smaller natural index;
* an item triggers when ``margin < threshold``;
* the sweep's objective is ``accuracy / (1 + 2 * f)``, where ``f`` is the
  fraction of items that trigger; among maximisers the **largest** threshold
  wins, so the criterion never spends an extra forward for nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def probabilities(logits: Sequence[float]) -> list[float]:
    """Softmax over the K code logits."""
    if not logits:
        raise ValueError("no logits")
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = math.fsum(exps)
    return [value / total for value in exps]


def ranked(logits: Sequence[float]) -> list[int]:
    """Natural option indices ordered by probability, ties by smaller index."""
    probs = probabilities(logits)
    return sorted(range(len(probs)), key=lambda index: (-probs[index], index))


def top_two(logits: Sequence[float]) -> tuple[int, int, float]:
    """The top two natural indices and their margin; requires K >= 2."""
    if len(logits) < 2:
        raise ValueError(f"need at least two options, got {len(logits)}")
    order = ranked(logits)
    probs = probabilities(logits)
    return order[0], order[1], probs[order[0]] - probs[order[1]]


def second_pass(candidates: Sequence[int], scores: dict[int, float]) -> int:
    """The judgement winner among ``candidates``; ties to the smaller index."""
    missing = [index for index in candidates if index not in scores]
    if missing:
        raise KeyError(f"no judgement score for {missing}")
    return min(candidates, key=lambda index: (-scores[index], index))


def prediction(
    base: int,
    candidates: Sequence[int],
    scores: dict[int, float],
    margin_value: float,
    threshold: float,
) -> tuple[int, bool]:
    """The cascade's answer and whether it spent the second pass."""
    if margin_value < threshold:
        return second_pass(candidates, scores), True
    return base, False


def evaluate(
    rows: Sequence[dict],
    threshold: float,
    *,
    eligible: set[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, trigger fraction and the objective at one threshold.

    ``rows`` carries ``id``, ``gold``, ``base``, ``candidates``, ``scores`` and
    ``margin`` per item.  A row with ``eligible: false`` never triggers and
    keeps the base answer, because the judgement readout cannot score it; the
    ``eligible`` argument overrides the rows when a subset is being evaluated.
    The trigger fraction is over every row either way.
    """
    correct = 0
    triggered = 0
    base_correct = 0
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        if eligible is None:
            item_eligible = bool(row.get("eligible", True))
        else:
            item_eligible = row["id"] in eligible
        answer, spent = prediction(
            row["base"],
            row["candidates"],
            row["scores"],
            row["margin"],
            threshold if item_eligible else 0.0,
        )
        correct += int(answer == row["gold"])
        base_correct += int(row["base"] == row["gold"])
        triggered += int(spent)
        cell = by_type.setdefault(
            row["type"],
            {"items": 0, "correct": 0, "base_correct": 0, "triggered": 0},
        )
        cell["items"] += 1
        cell["correct"] += int(answer == row["gold"])
        cell["base_correct"] += int(row["base"] == row["gold"])
        cell["triggered"] += int(spent)
    items = len(rows)
    fraction = triggered / items if items else 0.0
    accuracy = correct / items if items else 0.0
    return {
        "threshold": threshold,
        "items": items,
        "accuracy": accuracy,
        "correct": correct,
        "base_accuracy": base_correct / items if items else 0.0,
        "base_correct": base_correct,
        "triggered": triggered,
        "trigger_fraction": fraction,
        "mean_forwards": 1.0 + 2.0 * fraction,
        "objective": accuracy / (1.0 + 2.0 * fraction),
        "by_type": by_type,
    }


def sweep(
    rows: Sequence[dict],
    thresholds: Sequence[float],
    *,
    eligible: set[str] | None = None,
) -> list[dict[str, Any]]:
    """The curve at every threshold, in the order given."""
    return [evaluate(rows, threshold, eligible=eligible) for threshold in thresholds]


def threshold_grid(rows: Sequence[dict]) -> list[float]:
    """Every distinct margin, ascending, plus one value that never triggers.

    The margins are the complete set of distinct trigger sets a threshold can
    produce, so the sweep is exhaustive rather than sampled from a grid.
    """
    values = sorted({row["margin"] for row in rows})
    return [0.0, *values, 1.0]


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The sweep point with the highest objective; ties go to the largest threshold."""
    return max(rows, key=lambda row: (row["objective"], row["threshold"]))


def split_halves(ids: Sequence[str], mode: str) -> tuple[list[str], list[str]]:
    """Two disjoint, exhaustive halves of the ids; ``mode`` is ``even_odd`` or ``first_second``."""
    ordered = sorted(ids)
    if mode == "even_odd":
        return ordered[0::2], ordered[1::2]
    if mode == "first_second":
        middle = len(ordered) // 2
        return ordered[:middle], ordered[middle:]
    raise ValueError(f"unknown split mode {mode!r}")
