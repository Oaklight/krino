"""Calibration metrics: ECE, Brier score, reliability diagram data.

The metrics here are pure Python so they can be tested without torch. The
multiclass and ordinal helpers are what the decision-model evaluation needs on
top of the binary ECE/Brier pair: a choice question is scored against the whole
option vector, and a score question is ordinal, so a squared error on the rank
or a quadratic-weighted kappa says more than exact-match accuracy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


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


def multiclass_brier(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> dict[str, Any]:
    """Sum of squared errors over the full option vector, averaged over items.

    Unlike the binary Brier score this penalises probability assigned to the
    wrong options, not only to the gold one, so a confident wrong distribution
    costs more than a flat one.

    Args:
        probabilities: Probability vector per item, one per option.
        labels: Index of the gold option per item.

    Returns:
        ``{"brier": mean_over_items, "n": count}``.
    """
    if not probabilities:
        return {"brier": 0.0, "n": 0}
    total = 0.0
    for probs, label in zip(probabilities, labels):
        total += sum(
            (p - (1.0 if idx == label else 0.0)) ** 2
            for idx, p in enumerate(probs)
        )
    return {"brier": total / len(probabilities), "n": len(probabilities)}


def quadratic_weighted_kappa(
    predictions: Sequence[float],
    labels: Sequence[float],
    n_levels: int,
) -> float:
    """Quadratic-weighted kappa between two ordinal ratings.

    Weights the agreement by the squared distance between levels, so a
    prediction one level away is penalised far less than one three levels away.
    0.0 is chance agreement and 1.0 is perfect agreement.

    Args:
        predictions: Predicted level per item, on ``0 .. n_levels-1``.
        labels: Gold level per item, same scale.
        n_levels: Number of ordinal levels.

    Returns:
        The kappa, or 0.0 when the labels show no variance.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not predictions or n_levels < 2:
        return 0.0

    observed = [[0.0] * n_levels for _ in range(n_levels)]
    for pred, gold in zip(predictions, labels):
        observed[int(pred)][int(gold)] += 1
    total = float(len(predictions))

    hist_pred = [sum(row) / total for row in observed]
    hist_gold = [sum(observed[i][j] for i in range(n_levels)) / total for j in range(n_levels)]
    expected = [[hist_pred[i] * hist_gold[j] for j in range(n_levels)] for i in range(n_levels)]

    denom = (n_levels - 1) ** 2
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(n_levels):
        for j in range(n_levels):
            weight = (i - j) ** 2 / denom
            observed_disagreement += weight * observed[i][j] / total
            expected_disagreement += weight * expected[i][j]
    if expected_disagreement == 0.0:
        return 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a single logit vector."""
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for row, label in zip(logits, labels):
        probs = softmax([value / temperature for value in row])
        total -= math.log(max(probs[label], 1e-12))
    return total / max(len(logits), 1)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    low: float = 0.05,
    high: float = 20.0,
    steps: int = 400,
) -> dict[str, Any]:
    """Fit one temperature by minimising NLL on held-out logits.

    Temperature scaling is the cheapest calibration fix there is: it leaves the
    argmax untouched, so accuracy cannot change, and it usually moves ECE by a
    large factor. Fitting needs only a few hundred items.

    Args:
        logits: Logit vector per item.
        labels: Gold index per item.
        low: Lower bound of the search grid.
        high: Upper bound of the search grid.
        steps: Grid resolution before the local refinement.

    Returns:
        ``{"temperature": float, "nll_before": float, "nll_after": float,
        "n": count}``. ``nll_before`` uses T=1.
    """
    if not logits:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0, "n": 0}

    def search(lo: float, hi: float, count: int) -> tuple[float, float]:
        best_temp, best_nll = lo, float("inf")
        for step in range(count):
            temp = lo * (hi / lo) ** (step / (count - 1))
            value = _nll(logits, labels, temp)
            if value < best_nll:
                best_temp, best_nll = temp, value
        return best_temp, best_nll

    coarse_temp, _ = search(low, high, max(steps, 2))
    span = coarse_temp * (high / low) ** (1.0 / max(steps - 1, 1))
    fine_temp, fine_nll = search(max(coarse_temp / span, low), min(coarse_temp * span, high), 40)

    return {
        "temperature": fine_temp,
        "nll_before": _nll(logits, labels, 1.0),
        "nll_after": fine_nll,
        "n": len(logits),
    }


def top1_calibration(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = 10,
) -> dict[str, Any]:
    """ECE over top-1 confidence for multiclass predictions."""
    confidences = [max(row) for row in probabilities]
    correctness = [
        max(range(len(row)), key=row.__getitem__) == label
        for row, label in zip(probabilities, labels)
    ]
    return expected_calibration_error(confidences, correctness, n_bins)
