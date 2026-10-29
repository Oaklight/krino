"""Calibration of the routed choice system, measured from per-item score vectors.

The project's best overall accuracy comes from a rule, not from a single scorer:
an item with fewer than twenty options is answered by the backbone's native
readout and a longer one by the trained head (``research/17``). That rule is
what meets goal item 1, and its own calibration has never been measured — each
path calibrates under 0.05 separately, but pooling them is not the same number,
because the two paths produce probabilities of different sharpness.

This module builds the probability vector each item would be served, applies one
temperature per path, and reports ECE and Brier half-and-half: the temperature is
fitted on one half and scored on the other, then the halves swap. It is pure
Python so the arithmetic that decides the claim runs on a bare CI interpreter.

Input is the record format written by ``model/scripts/blend_scorers.py
--dump-records``: ``{"k", "gold", "head", "native"}`` per item, where ``head`` is
a probability vector and ``native`` is a vector of per-option log-odds, or
``None`` where the long-list path skipped it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

from model.evaluation.calibration import (
    expected_calibration_error,
    fit_temperature,
    multiclass_brier,
    softmax,
)

# The option count at which the rule stops using the native readout. This is the
# same boundary `model/evaluation/scorer_blend.py` calls LONG_MIN; it is repeated
# here rather than imported so that measuring the rule does not depend on the
# fusion code being present.
LONG_MIN = 20


def path_of(record: dict[str, Any]) -> str:
    """Which scorer answers this item under the deployed rule.

    Args:
        record: One dumped item.

    Returns:
        ``"native"`` for short lists with a native score, ``"head"`` otherwise.
        A short item that was dumped without its native score also falls back to
        the head, and ``routed_report`` counts how often that happens rather than
        hiding it.
    """
    native = record.get("native")
    if record["k"] < LONG_MIN and native:
        return "native"
    return "head"


def item_logits(record: dict[str, Any]) -> list[float]:
    """The logit vector the serving path produces for one item.

    Both vectors in the dump are already logits: the head is a linear head, so
    its output vector is used as it is, and the native vector is the per-option
    log-odds. The probabilities a reader sees anywhere below come from a softmax
    over these, with the path's temperature.

    Args:
        record: One dumped item.

    Returns:
        A logit vector aligned with the item's options.
    """
    if path_of(record) == "native":
        return [float(value) for value in record["native"]]
    return [float(value) for value in record["head"]]


def _served(
    records: Sequence[dict[str, Any]], temperatures: dict[str, float]
) -> list[dict[str, Any]]:
    """Probability vector, gold label and path for every item, at ``temperatures``."""
    served = []
    for record in records:
        path = path_of(record)
        scaled = [value / temperatures[path] for value in item_logits(record)]
        served.append(
            {
                "path": path,
                "probabilities": softmax(scaled),
                "label": int(record["gold"]),
                "k": int(record["k"]),
            }
        )
    return served


def _metrics(served: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """ECE and multiclass Brier over already-served items."""
    if not served:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0}
    probabilities = [item["probabilities"] for item in served]
    labels = [item["label"] for item in served]
    predictions = [
        max(range(len(row)), key=lambda index: (row[index], -index)) for row in probabilities
    ]
    correct = [guess == label for guess, label in zip(predictions, labels)]
    ece = expected_calibration_error(
        [max(row) for row in probabilities], correct
    )["ece"]
    return {
        "n": len(served),
        "accuracy": sum(correct) / len(correct),
        "ece": ece,
        "brier": multiclass_brier(probabilities, labels)["brier"],
    }


def _fit(records: Sequence[dict[str, Any]], criterion: str) -> dict[str, float]:
    """One temperature per path, fitted on these records by ``criterion``."""
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_path[path_of(record)].append(record)
    temperatures = {"native": 1.0, "head": 1.0}
    for path, rows in by_path.items():
        if not rows:
            continue
        fitted = fit_temperature(
            [item_logits(record) for record in rows],
            [int(record["gold"]) for record in rows],
            criterion=criterion,
        )
        temperatures[path] = fitted["temperature"]
    return temperatures


def routed_report(
    records: Sequence[dict[str, Any]],
    folds: int = 2,
    criterion: str = "ece",
    head_path: Callable[[dict[str, Any], int, int], int] | None = None,
) -> dict[str, Any]:
    """ECE and Brier of the routed system, cross-fitted over ``folds``.

    The temperature a path uses is fitted on the other folds' items, so the
    reported ECE is always on items that did not choose it — the same protocol
    ``eval_calibration.py`` uses with two halves, generalised to ``folds``.

    Args:
        records: Dumped items with ``k``, ``gold``, ``head`` and ``native``.
        folds: Number of cross-fitting folds, alternating by list position.
        criterion: ``"ece"`` (matches the goal's wording) or ``"nll"``.
        head_path: Optional replacement for :func:`path_of`, taking ``(record,
            index, n_folds)``; used to ask what the aggregate would be under a
            different choice of paths.

    Returns:
        ``{"paths", "folds": [...], "mean": {...}}`` where each fold holds the
        fitted temperatures, the head-only and routed metrics, and the routed
        metrics at ``T = 1`` for comparison.
    """
    if folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    records = list(records)
    if not records:
        raise ValueError("no records to report on")

    path_fn = head_path or (lambda record, index, count: path_of(record))
    if head_path is not None:
        substituted = []
        for index, record in enumerate(records):
            copy = dict(record)
            path = path_fn(record, index, folds)
            if path == "head":
                copy["native"] = None
            substituted.append(copy)
        records = substituted

    assignment = [index % folds for index in range(len(records))]
    counts = {path: 0 for path in ("native", "head")}
    for record in records:
        counts[path_of(record)] += 1

    fold_reports = []
    for fold in range(folds):
        fit_rows = [
            record for record, assigned in zip(records, assignment) if assigned != fold
        ]
        eval_rows = [
            record for record, assigned in zip(records, assignment) if assigned == fold
        ]
        temperatures = _fit(fit_rows, criterion)
        # The goal names per-type scaling, which is one temperature for the whole
        # question type, not one per path. Both are reported because the gap
        # between them is the cost of the deployed rule having two scorers.
        single_temperature = fit_temperature(
            [item_logits(record) for record in fit_rows],
            [int(record["gold"]) for record in fit_rows],
            criterion=criterion,
        )["temperature"]
        head_temperature = fit_temperature(
            [item_logits({**record, "native": None}) for record in fit_rows],
            [int(record["gold"]) for record in fit_rows],
            criterion=criterion,
        )["temperature"]
        served = _served(eval_rows, temperatures)
        head_only = _metrics(
            [
                {
                    "probabilities": softmax(item_logits({**record, "native": None})),
                    "label": int(record["gold"]),
                    "k": int(record["k"]),
                }
                for record in eval_rows
            ]
        )
        head_only_calibrated = _metrics(
            [
                {
                    "probabilities": softmax(
                        [
                            value / head_temperature
                            for value in item_logits({**record, "native": None})
                        ]
                    ),
                    "label": int(record["gold"]),
                    "k": int(record["k"]),
                }
                for record in eval_rows
            ]
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_n": len(fit_rows),
                "eval_n": len(eval_rows),
                "temperature": temperatures,
                "single_temperature": single_temperature,
                "head_temperature": head_temperature,
                "head_only": head_only,
                "head_only_calibrated": head_only_calibrated,
                "routed": _metrics(served),
                "routed_single_t": _metrics(
                    _served(
                        eval_rows,
                        {"native": single_temperature, "head": single_temperature},
                    )
                ),
                "routed_at_t1": _metrics(_served(eval_rows, {"native": 1.0, "head": 1.0})),
            }
        )

    def mean_of(metric: str, key: str) -> float:
        return sum(report[metric][key] for report in fold_reports) / len(fold_reports)

    return {
        "rule": {"long_min": LONG_MIN, "criterion": criterion, "folds": folds},
        "paths": counts,
        "folds": fold_reports,
        "mean": {
            metric: {key: mean_of(metric, key) for key in ("n", "accuracy", "ece", "brier")}
            for metric in (
                "head_only",
                "head_only_calibrated",
                "routed",
                "routed_single_t",
                "routed_at_t1",
            )
        },
    }


def by_path(records: Sequence[dict[str, Any]], temperatures: dict[str, float]) -> dict[str, Any]:
    """Metrics per path at fixed temperatures, for diagnosing the pooled number."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for served_item in _served(records, temperatures):
        groups[served_item["path"]].append(served_item)
    return {path: _metrics(items) for path, items in sorted(groups.items())}


def buckets(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Item counts by the option-count buckets the notes report on."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        k = int(record["k"])
        if k <= 5:
            counts["<=5"] += 1
        elif k < LONG_MIN:
            counts[f"6-{LONG_MIN - 1}"] += 1
        else:
            counts[f">={LONG_MIN}"] += 1
    return dict(sorted(counts.items()))
