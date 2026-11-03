"""Torch-free metrics for the Krino v3 evaluation gate.

One command must answer every acceptance question in one JSON: accuracy,
calibration, probability skill, ordinal agreement, option-count degradation,
schema holdout, abstention, and option-order robustness. This module holds the
arithmetic so it can be tested without torch; ``model/scripts/eval_gate.py``
runs the model and calls it.

A ``Prediction`` carries a probability vector aligned with ``keys``. For noul
the keys are ``"no"`` and ``"yes"``; for choice they are the criteria keys; for
score they are ``"0"`` through ``"L-1"``. Level labels are zero-based.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Prediction:
    """One scored item, in the form every metric below consumes."""

    item_id: str
    q_type: str
    label: Any
    probs: tuple[float, ...]
    keys: tuple[str, ...]
    source: str = ""
    schema: str = ""

    @property
    def label_index(self) -> int | None:
        """Index of the gold answer in ``keys``, or ``None`` if unmapped."""
        if self.q_type == "noul":
            return 1 if bool(self.label) else 0
        if self.q_type == "choice":
            label = str(self.label)
            return self.keys.index(label) if label in self.keys else None
        if self.q_type == "score":
            try:
                idx = int(round(float(self.label)))
            except (TypeError, ValueError):
                return None
            return idx if 0 <= idx < len(self.probs) else None
        return None

    @property
    def predicted_index(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def predicted_key(self) -> str:
        return self.keys[self.predicted_index]

    @property
    def confidence(self) -> float:
        return self.probs[self.predicted_index]

    @property
    def correct(self) -> bool:
        idx = self.label_index
        return idx is not None and idx == self.predicted_index


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Top-1 ECE with equal-width confidence bins."""
    total = len(confidences)
    if total == 0:
        return 0.0
    error = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        members = [
            j
            for j, conf in enumerate(confidences)
            if (low <= conf < high) or (i == n_bins - 1 and conf == high)
        ]
        if not members:
            continue
        avg_conf = sum(confidences[j] for j in members) / len(members)
        avg_acc = sum(1.0 if correct[j] else 0.0 for j in members) / len(members)
        error += abs(avg_acc - avg_conf) * len(members) / total
    return error


def binary_brier(prob_true: float, label: bool) -> float:
    """Brier score for a binary outcome, using P(true) only."""
    target = 1.0 if label else 0.0
    return (prob_true - target) ** 2


def multiclass_brier(probs: Sequence[float], label_index: int) -> float:
    """Sum of squared errors over the full probability vector."""
    return sum(
        (prob - (1.0 if j == label_index else 0.0)) ** 2
        for j, prob in enumerate(probs)
    )


def brier_skill_score(model_brier: float, reference_brier: float) -> float | None:
    """Brier skill against a reference forecast; ``None`` if the reference is degenerate."""
    if reference_brier <= 0:
        return None
    return 1.0 - model_brier / reference_brier


def quadratic_weighted_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_levels: int,
) -> float | None:
    """Quadratic weighted kappa over an ordinal scale with ``n_levels`` values."""
    if n_levels < 2 or not y_true:
        return None
    observed = [[0] * n_levels for _ in range(n_levels)]
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_levels and 0 <= pred < n_levels:
            observed[true][pred] += 1
    row_sums = [sum(observed[i]) for i in range(n_levels)]
    col_sums = [sum(observed[i][j] for i in range(n_levels)) for j in range(n_levels)]
    total = sum(row_sums)
    if total == 0:
        return None
    numerator = 0.0
    denominator = 0.0
    for i in range(n_levels):
        for j in range(n_levels):
            weight = ((i - j) ** 2) / ((n_levels - 1) ** 2)
            numerator += weight * observed[i][j]
            denominator += weight * row_sums[i] * col_sums[j] / total
    if denominator <= 0:
        return None
    return 1.0 - numerator / denominator


def rank_probability_score(probs: Sequence[float], label_index: int) -> float:
    """Rank probability score (normalised by the number of thresholds)."""
    n_levels = len(probs)
    if n_levels < 2:
        return 0.0
    cumulative_prob = 0.0
    cumulative_label = 0.0
    score = 0.0
    for k in range(n_levels - 1):
        cumulative_prob += probs[k]
        cumulative_label += 1.0 if k == label_index else 0.0
        score += (cumulative_prob - cumulative_label) ** 2
    return score / (n_levels - 1)


# ---------------------------------------------------------------------------
# Reference distributions and grouped summaries
# ---------------------------------------------------------------------------


def _reference_counts(
    rows: Sequence[Prediction],
) -> tuple[list[int], Counter[int], Counter[str]]:
    noul = [0, 0]
    score: Counter[int] = Counter()
    choice: Counter[str] = Counter()
    for row in rows:
        idx = row.label_index
        if idx is None:
            continue
        if row.q_type == "noul":
            noul[idx] += 1
        elif row.q_type == "score":
            score[idx] += 1
        elif row.q_type == "choice":
            choice[row.keys[idx]] += 1
    return noul, score, choice


def _reference_probs(
    row: Prediction,
    noul: list[int],
    score: Counter[int],
    choice: Counter[str],
) -> list[float]:
    """Reference forecast for one row, from the evaluated label marginal."""
    if row.q_type == "noul":
        total = sum(noul)
        if total == 0:
            return [0.5, 0.5]
        return [noul[0] / total, noul[1] / total]
    if row.q_type == "score":
        total = sum(score.values())
        if total == 0:
            return [1.0 / len(row.probs)] * len(row.probs)
        probs = [score.get(i, 0) / total for i in range(len(row.probs))]
        norm = sum(probs)
        return [p / norm for p in probs] if norm else [1.0 / len(probs)] * len(row.probs)
    weights = [choice.get(key, 0) for key in row.keys]
    total = sum(weights)
    if total == 0:
        return [1.0 / len(row.keys)] * len(row.keys)
    return [w / total for w in weights]


def summarize(rows: Iterable[Prediction], n_bins: int = 10) -> dict[str, Any]:
    """Accuracy, ECE, Brier, BSS, and score-only ordinal metrics for one group."""
    usable = [row for row in rows if row.label_index is not None]
    if not usable:
        return {"n": 0}

    noul, score_counts, choice_counts = _reference_counts(usable)
    correct: list[bool] = []
    confidences: list[float] = []
    model_briers: list[float] = []
    reference_briers: list[float] = []
    score_true: list[int] = []
    score_pred: list[int] = []
    score_mae: list[float] = []
    score_rps: list[float] = []

    for row in usable:
        label_idx = row.label_index
        assert label_idx is not None
        correct.append(row.correct)
        confidences.append(row.confidence)
        reference = _reference_probs(row, noul, score_counts, choice_counts)
        if row.q_type == "noul":
            # Binary Brier convention: (p_true - y)^2, not the 2x sum over
            # both classes. Choice and score use the multiclass vector.
            model_briers.append(binary_brier(row.probs[1], bool(row.label)))
            reference_briers.append(binary_brier(reference[1], bool(row.label)))
        else:
            model_briers.append(multiclass_brier(row.probs, label_idx))
            reference_briers.append(multiclass_brier(reference, label_idx))
        if row.q_type == "score":
            score_true.append(label_idx)
            score_pred.append(row.predicted_index)
            expected = sum(j * p for j, p in enumerate(row.probs))
            score_mae.append(abs(expected - float(row.label)))
            score_rps.append(rank_probability_score(row.probs, label_idx))

    brier = _mean(model_briers)
    result: dict[str, Any] = {
        "n": len(usable),
        "accuracy": _mean([1.0 if c else 0.0 for c in correct]),
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "brier": brier,
        "bss": brier_skill_score(brier, _mean(reference_briers)),
        "mean_confidence": _mean(confidences),
    }
    if score_rps:
        n_levels = max(len(row.probs) for row in usable if row.q_type == "score")
        result["qwk"] = quadratic_weighted_kappa(score_true, score_pred, n_levels)
        result["rps"] = _mean(score_rps)
        result["mae"] = _mean(score_mae)
    return result


def summarize_by(
    rows: Iterable[Prediction],
    key_fn: Callable[[Prediction], str],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Group rows by ``key_fn`` and summarize each group."""
    groups: dict[str, list[Prediction]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return {key: summarize(members, n_bins) for key, members in sorted(groups.items())}


def k_bucket(k: int) -> str:
    """Option-count bucket from the v3 calibration protocol."""
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 20:
        return "6-20"
    if k <= 80:
        return "21-80"
    if k <= 256:
        return "81-256"
    return ">256"


# ---------------------------------------------------------------------------
# Schema identity and the schema split
# ---------------------------------------------------------------------------


def schema_key(question: dict[str, Any]) -> str:
    """Stable identity for ``(instructions, label set)``.

    Instructions are case-folded and whitespace-collapsed so paraphrases that
    differ only in formatting group together. Choice keys are sorted; score
    levels are identified by position. The digest is truncated because it is a
    grouping key, not a security primitive.
    """
    instructions = " ".join(str(question.get("instructions", "")).lower().split())
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        keys = sorted(str(key) for key in criteria)
    elif isinstance(criteria, list):
        keys = [str(i) for i in range(len(criteria))]
    else:
        keys = []
    payload = json.dumps([instructions, keys], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def summarize_schema_split(
    rows: Iterable[Prediction],
    split_map: dict[str, str] | None,
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Summarise seen-schema versus held-out-schema, when a split is supplied."""
    if not split_map:
        return {}
    groups: dict[str, list[Prediction]] = {}
    for row in rows:
        groups.setdefault(split_map.get(row.schema, "unassigned"), []).append(row)
    return {key: summarize(members, n_bins) for key, members in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Robustness and selective prediction
# ---------------------------------------------------------------------------


def order_sensitivity(
    before: Sequence[Prediction],
    after: Sequence[Prediction],
) -> dict[str, Any]:
    """How much a presentation-order change moves the answer and the distribution."""
    before_by_id = {row.item_id: row for row in before}
    after_by_id = {row.item_id: row for row in after}
    changed = 0
    tv_distances: list[float] = []
    accuracy_after: list[bool] = []
    for item_id, original in before_by_id.items():
        permuted = after_by_id.get(item_id)
        if permuted is None:
            continue
        if original.predicted_key != permuted.predicted_key:
            changed += 1
        original_probs = dict(zip(original.keys, original.probs))
        permuted_probs = dict(zip(permuted.keys, permuted.probs))
        keys = set(original_probs) | set(permuted_probs)
        tv_distances.append(
            0.5 * sum(abs(original_probs.get(k, 0.0) - permuted_probs.get(k, 0.0)) for k in keys)
        )
        accuracy_after.append(permuted.correct)
    n = len(tv_distances)
    return {
        "n": n,
        "changed_fraction": changed / n if n else 0.0,
        "mean_tv": _mean(tv_distances),
        "accuracy_after": _mean([1.0 if c else 0.0 for c in accuracy_after]),
    }


def risk_coverage(
    rows: Sequence[Prediction],
    grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, Any]:
    """Selective accuracy when the least-confident items abstain first.

    This is the max-prob baseline the P3 answerable head must beat: no learned
    head is involved, only the top-1 confidence.
    """
    usable = [row for row in rows if row.label_index is not None]
    if not usable:
        return {"n": 0, "curve": [], "aurc": None, "risk_at_80": None}
    ordered = sorted(usable, key=lambda row: row.confidence, reverse=True)
    curve: list[dict[str, Any]] = []
    for coverage in grid:
        k = max(1, min(len(ordered), int(math.ceil(coverage * len(ordered)))))
        selected = ordered[:k]
        accuracy = _mean([1.0 if row.correct else 0.0 for row in selected])
        curve.append(
            {
                "coverage": coverage,
                "n": k,
                "selective_accuracy": accuracy,
                "risk": 1.0 - accuracy,
            }
        )
    risks = [point["risk"] for point in curve]
    risk_at_80 = next(
        (point["risk"] for point in curve if point["coverage"] >= 0.8 - 1e-9), None
    )
    return {
        "n": len(usable),
        "curve": curve,
        "aurc": _mean(risks),
        "risk_at_80": risk_at_80,
    }
