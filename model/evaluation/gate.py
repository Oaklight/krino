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
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

from model.evaluation.paired_power import detectable_delta


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
    tier: str = ""

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
                level = int(round(float(self.label)))
            except (TypeError, ValueError):
                return None
            # Score keys are level numbers.  Resolve through them so a row
            # presented in permuted order (``predict_reversed``) still finds
            # its gold level; a positional read scored every correct reversed
            # score item as wrong.  In natural order keys[i] == str(i), so this
            # is the same index as before.
            if self.keys and all(key.lstrip("-").isdigit() for key in self.keys):
                key = str(level)
                return self.keys.index(key) if key in self.keys else None
            return level if 0 <= level < len(self.probs) else None
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


def chance_corrected_accuracy(rows: Iterable[Prediction]) -> dict[str, Any]:
    """Chance-corrected accuracy, the local Intelligence proxy.

    Each item is normalised by its own chance level ``c = 1 / n_options``:
    ``(correct - c) / (1 - c)``.  A model that always guesses scores 0, a
    perfect model scores 1, and an item with one option is skipped because it
    carries no decision.  The JevBench tier weights are not applied here
    because the standard, judge and sealed tiers are not public.

    ``by_options`` holds the item and correct counts per option count.  The
    correction is linear in ``correct`` within one option count, so these
    counts reproduce ``corrected`` exactly; the mean ``chance`` alone does not.
    """
    raw: list[float] = []
    corrected: list[float] = []
    chances: list[float] = []
    by_options: dict[int, dict[str, int]] = {}
    for row in rows:
        n_options = len(row.probs)
        if n_options < 2:
            continue
        chance = 1.0 / n_options
        correct = 1.0 if row.predicted_index == row.label_index else 0.0
        raw.append(correct)
        chances.append(chance)
        corrected.append((correct - chance) / (1.0 - chance))
        cell = by_options.setdefault(n_options, {"n": 0, "correct": 0})
        cell["n"] += 1
        cell["correct"] += int(correct)
    if not corrected:
        return {"n": 0, "accuracy": 0.0, "chance": 0.0, "corrected": 0.0, "by_options": {}}
    return {
        "n": len(corrected),
        "accuracy": sum(raw) / len(raw),
        "chance": sum(chances) / len(chances),
        "corrected": sum(corrected) / len(corrected),
        "by_options": {str(k): by_options[k] for k in sorted(by_options)},
    }


def corrected_from_counts(by_options: dict[str, dict[str, int]]) -> float:
    """Recompute chance-corrected accuracy from ``by_options`` counts."""
    total = 0.0
    items = 0
    for key, cell in by_options.items():
        chance = 1.0 / int(key)
        total += (cell["correct"] - cell["n"] * chance) / (1.0 - chance)
        items += cell["n"]
    return total / items if items else 0.0


# ---------------------------------------------------------------------------
# Paired gate screen (#263, policy in research/37-gate-policy.md)
# ---------------------------------------------------------------------------

MATERIAL_REGRESSION_SUPPORTED = "material regression supported"
COARSE_SAFETY_SCREEN_PASSED = "coarse safety screen passed"
UNRESOLVED = "unresolved"
IMPROVEMENT_SUPPORTED = "improvement supported"
IMPROVEMENT_SUGGESTED_BUT_UNRESOLVED = "improvement suggested but unresolved"

#: JevBench coarse margin: 8 pp of chance-corrected accuracy. It is the
#: instrument's resolution at 231 items, not a scientific tolerance; a genuine
#: 6 pp regression passes this screen. See ``research/37-gate-policy.md``.
JEVBENCH_MARGIN_PP = 8.0


@dataclass(frozen=True)
class PairedOutcome:
    """One item's outcome under one readout, with the item's chance level."""

    correct: bool
    n_options: int

    def __post_init__(self) -> None:
        if self.n_options < 2:
            raise ValueError(f"n_options must be at least 2, got {self.n_options}")


@dataclass(frozen=True)
class GateScreen:
    """The policy's output for one arm-against-control comparison.

    ``states`` always carries exactly one of the three screen states and may
    carry one improvement state as well.  The screen states are assigned from
    the confidence interval, never from the point estimate alone.
    """

    n: int
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    wins: int
    losses: int
    p_value: float
    discordance: float
    detectable_pp: float | None
    states: frozenset[str]
    action: str
    reason: str


def _exact_p(
    arm: Mapping[str, PairedOutcome],
    control: Mapping[str, PairedOutcome],
    ids: Sequence[str],
) -> float:
    """Exact two-sided McNemar p, from the form the gate has always used.

    Imported here, not at module scope, so the evaluation layer does not take
    a module-level dependency on the comparison CLI that prints this result.
    """
    from model.scripts.paired_mcnemar import mcnemar

    first = {item_id: arm[item_id].correct for item_id in ids}
    second = {item_id: control[item_id].correct for item_id in ids}
    return mcnemar(first, second, list(ids))[2]


def paired_screen(
    arm: Mapping[str, PairedOutcome],
    control: Mapping[str, PairedOutcome],
    ids: Iterable[str] | None = None,
    *,
    margin_pp: float = JEVBENCH_MARGIN_PP,
    alpha: float = 0.05,
) -> GateScreen:
    """Three-state interval screen for an arm against its control.

    The paired effect is the difference in chance-corrected accuracy: each
    item's signed difference is ``(arm - control) / (1 - 1 / n_options)``.
    The 95% interval is the normal approximation on those per-item differences,
    ``mean +/- z * sd / sqrt(n)`` with ``z`` from the two-sided ``alpha``.  The
    exact two-sided McNemar p-value is reported beside it, and the detectable
    difference at 80% power comes from ``model.evaluation.paired_power`` using
    the measured discordance.

    The screen states are assigned by the interval: material regression
    supported when the whole interval is below ``-margin_pp``, coarse safety
    screen passed when it is entirely above ``-margin_pp``, and unresolved when
    it crosses.  A point estimate below the margin is not evidence of a
    regression on its own.  Improvement states use the same interval: supported
    when its lower bound is above zero, suggested but unresolved when the point
    estimate is positive and the interval crosses zero.

    The result also carries the policy's recommended action and a one-sentence
    reason for the screen state.  The action is advice with the evidence
    attached, not a verdict to rubber-stamp.

    Raises:
        ValueError: If fewer than two items are shared, if ``alpha`` is outside
            ``(0, 1)``, or if an item carries a different option count in the
            two maps.
        KeyError: If a requested id is missing from either map.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if ids is None:
        candidates = sorted(set(arm) & set(control))
    else:
        candidates = list(ids)
    ordered: list[str] = []
    seen: set[str] = set()
    for item_id in candidates:
        if item_id in seen:
            continue
        seen.add(item_id)
        if item_id not in arm or item_id not in control:
            raise KeyError(f"{item_id}: missing from arm or control outcomes")
        ordered.append(item_id)
    n = len(ordered)
    if n < 2:
        raise ValueError(f"paired screen needs at least two shared items, got {n}")

    diffs: list[float] = []
    wins = 0
    losses = 0
    for item_id in ordered:
        arm_row = arm[item_id]
        control_row = control[item_id]
        if arm_row.n_options != control_row.n_options:
            raise ValueError(
                f"{item_id}: arm has {arm_row.n_options} options, "
                f"control has {control_row.n_options}"
            )
        weight = 1.0 / (1.0 - 1.0 / arm_row.n_options)
        diffs.append(weight * (float(arm_row.correct) - float(control_row.correct)))
        if arm_row.correct and not control_row.correct:
            wins += 1
        elif control_row.correct and not arm_row.correct:
            losses += 1

    mean = sum(diffs) / n
    variance = sum((value - mean) ** 2 for value in diffs) / (n - 1)
    standard_error = math.sqrt(variance / n)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    low = mean - z * standard_error
    high = mean + z * standard_error

    discordance = (wins + losses) / n
    detectable = None
    if discordance > 0.0:
        detectable = detectable_delta(n, discordance) * 100.0

    states: set[str] = set()
    if high * 100.0 < -margin_pp:
        states.add(MATERIAL_REGRESSION_SUPPORTED)
        action = "block promotion"
        reason = (
            f"the 95% interval lies below -{margin_pp:g} pp, so the curated set "
            "supports a material regression"
        )
    elif low * 100.0 > -margin_pp:
        states.add(COARSE_SAFETY_SCREEN_PASSED)
        action = "no objection from the coarse screen"
        reason = (
            f"the 95% interval lies above -{margin_pp:g} pp; this is the "
            "instrument's resolution, not non-inferiority"
        )
    else:
        states.add(UNRESOLVED)
        action = "investigate"
        reason = (
            f"the 95% interval crosses -{margin_pp:g} pp; this item count cannot "
            "resolve a difference of this size"
        )
    if low > 0.0:
        states.add(IMPROVEMENT_SUPPORTED)
    elif mean > 0.0:
        states.add(IMPROVEMENT_SUGGESTED_BUT_UNRESOLVED)

    return GateScreen(
        n=n,
        delta_pp=mean * 100.0,
        ci_low_pp=low * 100.0,
        ci_high_pp=high * 100.0,
        wins=wins,
        losses=losses,
        p_value=_exact_p(arm, control, ordered),
        discordance=discordance,
        detectable_pp=detectable,
        states=frozenset(states),
        action=action,
        reason=reason,
    )


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
    accuracy_before: list[bool] = []
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
        # An unmapped gold label is not a wrong answer; keep it out of both
        # accuracies so they cover the same items and can be compared.
        if original.label_index is not None and permuted.label_index is not None:
            accuracy_before.append(original.correct)
            accuracy_after.append(permuted.correct)
    n = len(tv_distances)
    return {
        "n": n,
        "changed_fraction": changed / n if n else 0.0,
        "mean_tv": _mean(tv_distances),
        "n_scored": len(accuracy_after),
        "accuracy_before": _mean([1.0 if c else 0.0 for c in accuracy_before]),
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
