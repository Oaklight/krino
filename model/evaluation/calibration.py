"""Calibration metrics: ECE, Brier score, reliability diagram data."""

from __future__ import annotations

import math
from typing import Any


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> dict[str, Any]:
    if not confidences:
        return {"ece": 0.0, "n": 0, "bins": []}
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = [(low <= c < high) or (i == n_bins - 1 and c == high) for c in confidences]
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_correct = [cor for cor, m in zip(correctness, mask) if m]
        if bin_confs:
            avg_conf = sum(bin_confs) / len(bin_confs)
            avg_acc = sum(bin_correct) / len(bin_correct)
            bins.append({
                "bin": i,
                "range": (low, high),
                "count": len(bin_confs),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "gap": abs(avg_acc - avg_conf),
            })
        else:
            bins.append({"bin": i, "range": (low, high), "count": 0})
    total = len(confidences)
    ece = sum(b.get("gap", 0) * b["count"] / total for b in bins if b["count"] > 0)
    return {"ece": ece, "n": total, "n_bins": n_bins, "bins": bins}


def brier_score(probabilities: list[float], outcomes: list[bool]) -> dict[str, Any]:
    if not probabilities:
        return {"brier": 0.0, "n": 0}
    scores = [(p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probabilities, outcomes)]
    return {"brier": sum(scores) / len(scores), "n": len(scores)}


def choice_calibration(
    predicted_probs: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> dict[str, Any]:
    return expected_calibration_error(predicted_probs, correct, n_bins)


def reliability_diagram_data(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 15,
) -> list[dict[str, Any]]:
    result = expected_calibration_error(confidences, correctness, n_bins)
    return [b for b in result["bins"] if b["count"] > 0]
