"""Class-prior correction for an ordinal head, and the QWK aggregation it feeds.

A temperature and a monotone confidence map both leave an argmax untouched, so
neither can repair a head that under-predicts a level — only a change that moves
mass between levels can. Logit adjustment is the cheapest such change: shift
every level by a constant, ``alpha * log(p_reference / p_observed)``, and take
the argmax of the shifted vector.

Nothing here knows what a model is. The functions are pure so the arithmetic can
be tested without torch, and so the correction is applied identically to a
checkpoint and to a hand-written vector in a test.

The distinction that matters for interpretation is which marginal is the
reference:

* a *train* marginal is deployable — it is known before any eval item is seen;
* a *gold* marginal is an oracle — it is a bound on what any prior correction
  could achieve on those items, not a method.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from model.evaluation.calibration import quadratic_weighted_kappa

# A scale with fewer items than this has a QWK whose sampling noise dominates
# it; aggregates drop such scales instead of letting a handful of items move
# the mean.
MIN_SCALE_ITEMS = 30

# Reference probabilities are floored before the logarithm so a level the
# reference never shows produces a large finite shift rather than an infinite
# one and a NaN argmax.
_PROB_FLOOR = 1e-9


def marginal(indices: Sequence[int], n_levels: int) -> list[float]:
    """Empirical distribution of level indices over ``0 .. n_levels-1``.

    Args:
        indices: Level index per item.
        n_levels: Size of the level scale. Levels never seen get 0.0.

    Returns:
        A probability per level, summing to 1. An empty input returns all zeros
        rather than dividing by zero.
    """
    counts = [0] * n_levels
    for index in indices:
        if not 0 <= index < n_levels:
            raise ValueError(f"level index {index} outside 0..{n_levels - 1}")
        counts[index] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * n_levels
    return [count / total for count in counts]


def log_prior_shift(
    p_observed: Sequence[float], p_reference: Sequence[float], alpha: float
) -> list[float]:
    """Per-level additive shift that moves ``p_observed`` onto ``p_reference``.

    Args:
        p_observed: Marginal the head actually produces.
        p_reference: Marginal the correction aims at.
        alpha: Scale on the shift. 0 is the identity, 1 is the full correction.

    Returns:
        One shift per level, in logit units.

    Raises:
        ValueError: If the two marginals have different lengths.
    """
    if len(p_observed) != len(p_reference):
        raise ValueError("marginals must have the same length")
    return [
        alpha * (math.log(max(ref, _PROB_FLOOR)) - math.log(max(obs, _PROB_FLOOR)))
        for obs, ref in zip(p_observed, p_reference)
    ]


def adjusted_argmax(
    logits: Sequence[float],
    p_observed: Sequence[float],
    p_reference: Sequence[float],
    alpha: float,
) -> int:
    """Argmax of ``logits + shift``, ties broken toward the lower level.

    Ties are broken by level index rather than by position, for the same reason
    ``tie_stable_argmax`` exists in ``order_sensitivity``: otherwise the answer
    depends on where a level sat in the list. Here the levels are always in
    canonical order, but the rule is stated so the sweep cannot drift.

    Args:
        logits: Raw score-head logits, one per level.
        p_observed: Marginal the head produces on the items being corrected.
        p_reference: Marginal to correct toward.
        alpha: Scale on the correction.

    Returns:
        Index of the chosen level.
    """
    shift = log_prior_shift(p_observed, p_reference, alpha)
    shifted = [logit + delta for logit, delta in zip(logits, shift)]
    return max(range(len(shifted)), key=lambda index: (shifted[index], -index))


def qwk_by_scale(
    predictions_by_scale: dict[int, list[int]],
    gold_by_scale: dict[int, list[int]],
    min_items: int = MIN_SCALE_ITEMS,
) -> dict[str, Any]:
    """QWK per level scale, plus an item-weighted mean over readable scales.

    QWK is defined within one level scale. Averaging across scales produces a
    number that belongs to no scale, so the per-scale values are the finding and
    the mean is reported only because a single headline number is often asked
    for. Scales below ``min_items`` are excluded from both the mean and the
    accuracy and are named in ``excluded_scales``.

    Args:
        predictions_by_scale: Predicted level per item, keyed by level count.
        gold_by_scale: Gold level per item, keyed by the same counts.
        min_items: Smallest scale that may enter the aggregate.

    Returns:
        ``{"by_scale": {...}, "mixed_scale_qwk", "mixed_scale_accuracy",
        "excluded_scales"}``.

    Raises:
        ValueError: If the two mappings cover different scales or an item count
            disagrees.
    """
    if set(predictions_by_scale) != set(gold_by_scale):
        raise ValueError("predictions and gold cover different level counts")

    by_scale: dict[str, Any] = {}
    for n_levels in sorted(gold_by_scale):
        predictions = predictions_by_scale[n_levels]
        golds = gold_by_scale[n_levels]
        if len(predictions) != len(golds):
            raise ValueError(f"{n_levels}-level scale has {len(predictions)} predictions and {len(golds)} golds")
        if not golds:
            continue
        accuracy = sum(1 for guess, gold in zip(predictions, golds) if guess == gold) / len(golds)
        by_scale[str(n_levels)] = {
            "n": len(golds),
            "accuracy": accuracy,
            "qwk": quadratic_weighted_kappa(predictions, golds, n_levels),
        }

    readable = {level: group for level, group in by_scale.items() if group["n"] >= min_items}
    readable_n = sum(group["n"] for group in readable.values())
    return {
        "by_scale": by_scale,
        "mixed_scale_qwk": (
            sum(group["qwk"] * group["n"] for group in readable.values()) / readable_n
            if readable_n
            else 0.0
        ),
        "mixed_scale_accuracy": (
            sum(group["accuracy"] * group["n"] for group in readable.values()) / readable_n
            if readable_n
            else 0.0
        ),
        "excluded_scales": sorted(set(by_scale) - set(readable)),
    }
