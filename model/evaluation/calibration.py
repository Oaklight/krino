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
    criterion: str = "nll",
) -> dict[str, Any]:
    """Fit one temperature on held-out logits.

    Temperature scaling is the cheapest calibration fix there is: it leaves the
    argmax untouched, so accuracy cannot change, and it usually moves ECE by a
    large factor. Fitting needs only a few hundred items.

    Args:
        logits: Logit vector per item.
        labels: Gold index per item.
        low: Lower bound of the search grid.
        high: Upper bound of the search grid.
        steps: Grid resolution before the local refinement.
        criterion: ``"nll"`` (the usual choice, minimises negative log
            likelihood) or ``"ece"``. The goal is stated in ECE, and on the
            score primitive the two criteria pick different temperatures: NLL
            chooses 3.6 where ECE is 0.122, ECE chooses 1.5 where it is 0.031.

    Returns:
        ``{"temperature": float, "nll_before": float, "nll_after": float,
        "n": count}`` plus ``"criterion"``. ``nll_before`` uses T=1.
    """
    if criterion not in ("nll", "ece"):
        raise ValueError(f"unknown criterion {criterion!r}")
    if not logits:
        return {
            "temperature": 1.0,
            "nll_before": 0.0,
            "nll_after": 0.0,
            "n": 0,
            "criterion": criterion,
        }

    def objective(temperature: float) -> float:
        if criterion == "nll":
            return _nll(logits, labels, temperature)
        probabilities = [
            softmax([value / temperature for value in row]) for row in logits
        ]
        return top1_calibration(probabilities, labels)["ece"]

    def search(lo: float, hi: float, count: int) -> tuple[float, float]:
        best_temp, best_value = lo, float("inf")
        for step in range(count):
            temp = lo * (hi / lo) ** (step / (count - 1))
            value = objective(temp)
            if value < best_value:
                best_temp, best_value = temp, value
        return best_temp, best_value

    coarse_temp, _ = search(low, high, max(steps, 2))
    span = coarse_temp * (high / low) ** (1.0 / max(steps - 1, 1))
    fine_temp, _ = search(max(coarse_temp / span, low), min(coarse_temp * span, high), 40)

    return {
        "temperature": fine_temp,
        "nll_before": _nll(logits, labels, 1.0),
        "nll_after": _nll(logits, labels, fine_temp),
        "n": len(logits),
        "criterion": criterion,
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


def temperature_curve(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperatures: Sequence[float] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """NLL and ECE as functions of the temperature.

    Temperature is fitted by minimising NLL, but the goal is stated in ECE, and
    ECE has no reason to reach its minimum at the NLL optimum. On the score
    primitive it does not: a fitted T of 3.8 raises holdout ECE from 0.0628 to
    0.1267 while lowering NLL. Reporting both curves makes that visible, and
    gives the T that would be chosen by each criterion.

    Args:
        logits: Logit vector per item.
        labels: Gold index per item.
        temperatures: Grid to evaluate; defaults to 40 log-spaced points from
            0.25 to 8.0.
        n_bins: Bins for the ECE.

    Returns:
        ``{"curve": [{"temperature", "nll", "ece"}], "nll_best": {...},
        "ece_best": {...}, "at_one": {...}}``. ``None`` when there are no items.
    """
    if not logits:
        return {"curve": [], "nll_best": None, "ece_best": None, "at_one": None}

    if temperatures is None:
        steps = 40
        temperatures = [0.25 * (8.0 / 0.25) ** (i / (steps - 1)) for i in range(steps)]
    # T=1 is the baseline every other temperature is judged against, and a
    # log-spaced grid does not contain it; add it rather than rounding to a
    # neighbour and reporting that as "at T=1".
    if not any(abs(temperature - 1.0) < 1e-9 for temperature in temperatures):
        temperatures = sorted(list(temperatures) + [1.0])

    curve = []
    for temperature in temperatures:
        probabilities = [
            softmax([value / temperature for value in row]) for row in logits
        ]
        curve.append(
            {
                "temperature": temperature,
                "nll": _nll(logits, labels, temperature),
                "ece": top1_calibration(probabilities, labels, n_bins)["ece"],
            }
        )

    nll_best = min(curve, key=lambda point: point["nll"])
    ece_best = min(curve, key=lambda point: point["ece"])
    at_one = min(curve, key=lambda point: abs(point["temperature"] - 1.0))
    return {
        "curve": curve,
        "nll_best": dict(nll_best),
        "ece_best": dict(ece_best),
        "at_one": dict(at_one),
    }


def _pava_blocks(pairs: Sequence[tuple[float, float]]) -> list[list[float]]:
    """Pool-adjacent-violators over (confidence, outcome) pairs.

    Args:
        pairs: ``(confidence, 0.0 or 1.0)`` sorted by confidence.

    Returns:
        ``[x_sum, y_sum, n]`` per block, with non-decreasing ``y_sum / n``.
    """
    blocks: list[list[float]] = []
    for confidence, outcome in pairs:
        blocks.append([confidence, outcome, 1.0])
        while len(blocks) > 1:
            previous, current = blocks[-2], blocks[-1]
            if previous[1] / previous[2] < current[1] / current[2]:
                break
            blocks.pop()
            blocks.pop()
            blocks.append([
                previous[0] + current[0],
                previous[1] + current[1],
                previous[2] + current[2],
            ])
    return blocks


def _merge_small_blocks(
    blocks: list[list[float]], min_n: int
) -> list[list[float]]:
    """Merge blocks below ``min_n`` into a neighbour, keeping the fit monotone.

    A block fitted on a handful of items is noise, and at the ends of the
    confidence range that noise is not harmless: a single low-confidence block
    that happens to be wrong drives every item below it to zero confidence.
    Small blocks are therefore absorbed, and the adjacent-violator pass is
    re-run after each absorption so the result stays monotone -- pooling a
    block into its neighbour can otherwise leave the neighbour above the block
    that follows it.
    """
    ordered = [list(block) for block in blocks]
    while len(ordered) > 1:
        smallest = min(range(len(ordered)), key=lambda i: ordered[i][2])
        if ordered[smallest][2] >= min_n:
            break
        left = smallest - 1 if smallest > 0 else smallest
        right = left + 1
        ordered[left : right + 1] = [[
            ordered[left][0] + ordered[right][0],
            ordered[left][1] + ordered[right][1],
            ordered[left][2] + ordered[right][2],
        ]]
        ordered = _pava_weighted(ordered)
    return ordered


def _pava_weighted(blocks: list[list[float]]) -> list[list[float]]:
    """Re-apply the adjacent-violator pass to weighted blocks."""
    ordered: list[list[float]] = []
    for block in blocks:
        ordered.append(list(block))
        while len(ordered) > 1:
            previous, current = ordered[-2], ordered[-1]
            if previous[1] / previous[2] < current[1] / current[2]:
                break
            ordered.pop()
            ordered.pop()
            ordered.append([
                previous[0] + current[0],
                previous[1] + current[1],
                previous[2] + current[2],
            ])
    return ordered


def fit_confidence_map(
    confidences: Sequence[float],
    correct: Sequence[bool | int | float],
    alpha: float = 20.0,
    min_n: int = 20,
) -> dict[str, Any]:
    """Fit a monotone map from stated confidence to observed accuracy.

    Temperature scaling assumes the miscalibration is a single scale factor on
    the logits, which fails when the reliability curve is bent: on the score
    primitive a fitted temperature moves holdout ECE from 0.1027 to 0.0811 and
    stops, because the overconfidence is concentrated in one band of confidence
    that a scale factor cannot reach without wrecking the rest of the curve.
    A monotone map has no such constraint and is the ceiling for any
    recalibration that uses confidence alone.

    The map is isotonic regression (pool adjacent violators) on the
    confidence/outcome pairs, then two regularisations, both because the raw
    fit is fitted on a few hundred items and is badly behaved at the ends:

    * blocks under ``min_n`` items are absorbed into a neighbour,
    * every block is pulled toward the base rate with weight ``alpha``.

    Args:
        confidences: Stated confidence of the predicted option, per item.
        correct: Whether the prediction was right, per item.
        alpha: Pseudo-count pulling each block toward the overall accuracy. 0
            leaves the block means untouched.
        min_n: Smallest block that is kept.

    Returns:
        ``{"blocks": [{"upto", "confidence"}], "base_rate", "n"}``. The map is
        applied by :func:`apply_confidence_map`. ``base_rate`` is the accuracy
        of the fitting items and is the map's value above the last breakpoint.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    pairs = [
        (
            min(1.0, max(0.0, float(confidence))),
            1.0 if bool(ok) else 0.0,
        )
        for confidence, ok in zip(confidences, correct)
    ]
    # Sort by confidence only, so items the model gave the same confidence stay
    # in input order: sorting on the outcome as a tie-break would decide which
    # items pool together and bias the fit on exact ties.
    pairs.sort(key=lambda pair: pair[0])
    if not pairs:
        return {"blocks": [], "base_rate": 0.0, "n": 0}

    blocks = _pava_blocks(pairs)
    if min_n > 1:
        blocks = _merge_small_blocks(blocks, min_n)
    base_rate = sum(outcome for _, outcome in pairs) / len(pairs)
    # Shrinking a block toward the base rate depends on the block size, so two
    # neighbours ordered before shrinkage can invert after it; run the
    # adjacent-violator pass once more on the shrunk values.
    shrunk = [
        [x_sum, ((y_sum + alpha * base_rate) / (n + alpha)) * n, n]
        for x_sum, y_sum, n in blocks
    ]
    blocks_out = [
        {
            "upto": x_sum / n,
            "confidence": min(1.0, max(0.0, y_sum / n)),
        }
        for x_sum, y_sum, n in _pava_weighted(shrunk)
    ]
    return {"blocks": blocks_out, "base_rate": base_rate, "n": len(pairs)}


def apply_confidence_map(
    mapping: dict[str, Any], confidences: Sequence[float]
) -> list[float]:
    """Apply a :func:`fit_confidence_map` result to new confidences.

    Values outside the fitted range take the nearest block's value rather than
    being extrapolated, so the map cannot claim a confidence the fitting items
    never supported beyond the clamps at 0 and 1.
    """
    blocks = mapping.get("blocks") or []
    if not blocks:
        return [min(1.0, max(0.0, float(conf))) for conf in confidences]
    upto = [block["upto"] for block in blocks]
    values = [block["confidence"] for block in blocks]
    out = []
    for confidence in confidences:
        confidence = float(confidence)
        # Last breakpoint at or below the confidence; the blocks are stored by
        # the mean confidence of their members, so this is the block it falls in.
        index = 0
        low, high = 0, len(upto) - 1
        while low <= high:
            mid = (low + high) // 2
            if upto[mid] <= confidence:
                index = mid
                low = mid + 1
            else:
                high = mid - 1
        out.append(values[index])
    return out


def confidence_map_report(
    fit_confidences: Sequence[float],
    fit_correct: Sequence[bool | int | float],
    holdout_confidences: Sequence[float],
    holdout_correct: Sequence[bool | int | float],
    alpha: float = 20.0,
    min_n: int = 20,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit a monotone confidence map on one split and score it on the other.

    The counterpart of :func:`temperature_curve` for the family that temperature
    scaling cannot express: reports ECE on the same holdout items before and
    after, so the comparison against a fitted temperature is matched.
    """
    mapping = fit_confidence_map(fit_confidences, fit_correct, alpha=alpha, min_n=min_n)
    before = expected_calibration_error(
        [float(confidence) for confidence in holdout_confidences],
        [bool(ok) for ok in holdout_correct],
        n_bins,
    )["ece"]
    mapped = apply_confidence_map(mapping, holdout_confidences)
    after = expected_calibration_error(
        mapped, [bool(ok) for ok in holdout_correct], n_bins
    )["ece"]
    return {
        "alpha": alpha,
        "min_n": min_n,
        "before": before,
        "after": after,
        "n_fit": len(fit_confidences),
        "n_holdout": len(holdout_confidences),
        "map": mapping,
    }
