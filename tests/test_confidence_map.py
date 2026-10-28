"""Tests for the monotone confidence map (isotonic recalibration)."""

from __future__ import annotations

import random

import pytest

from model.evaluation.calibration import (
    apply_confidence_map,
    confidence_map_report,
    fit_confidence_map,
)


def overconfident(n: int = 2000, seed: int = 0) -> tuple[list[float], list[bool]]:
    """Stated confidence that is too high: a 0.8-confidence item is right 0.55 of the time."""
    rng = random.Random(seed)
    confidences = []
    correct = []
    for _ in range(n):
        confidence = rng.uniform(0.2, 0.95)
        # accuracy rises with confidence, but sits well below it
        accuracy = max(0.0, min(1.0, confidence - 0.25))
        confidences.append(confidence)
        correct.append(rng.random() < accuracy)
    return confidences, correct


def test_empty_input_returns_an_identity_map():
    mapping = fit_confidence_map([], [])
    assert mapping == {"blocks": [], "base_rate": 0.0, "n": 0}
    assert apply_confidence_map(mapping, [0.3, 0.9]) == [0.3, 0.9]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        fit_confidence_map([0.5], [])


def test_map_is_monotone_and_preserves_the_item_count():
    confidences, correct = overconfident()
    mapping = fit_confidence_map(confidences, correct, alpha=10.0, min_n=20)
    blocks = mapping["blocks"]
    assert mapping["n"] == len(confidences)
    assert blocks, "the fit produced no blocks"
    values = [block["confidence"] for block in blocks]
    assert values == sorted(values), values
    uptos = [block["upto"] for block in blocks]
    assert all(a < b for a, b in zip(uptos, uptos[1:])), uptos


def test_overconfidence_is_pulled_down_where_it_is_observed():
    confidences, correct = overconfident()
    mapping = fit_confidence_map(confidences, correct, alpha=20.0, min_n=20)
    mapped = apply_confidence_map(mapping, [0.8])
    assert mapped[0] < 0.7, mapped


def test_calibrated_data_is_left_nearly_alone():
    """When accuracy matches confidence, the map should not move it much."""
    rng = random.Random(7)
    confidences = []
    correct = []
    for _ in range(4000):
        confidence = rng.uniform(0.3, 0.95)
        confidences.append(confidence)
        correct.append(rng.random() < confidence)
    mapping = fit_confidence_map(confidences, correct, alpha=20.0, min_n=20)
    for probe in (0.4, 0.6, 0.8):
        mapped = apply_confidence_map(mapping, [probe])[0]
        assert abs(mapped - probe) < 0.08, (probe, mapped)


def test_min_n_absorbs_small_blocks_without_losing_items():
    # Two hundred items at 0.9 confidence plus three at 0.05, all wrong: the
    # three are too few to support a block of their own.
    confidences = [0.9] * 200 + [0.05] * 3
    correct = [True] * 180 + [False] * 20 + [False] * 3
    mapping = fit_confidence_map(confidences, correct, alpha=0.0, min_n=20)
    sizes = [block["upto"] for block in mapping["blocks"]]
    assert len(mapping["blocks"]) >= 1
    assert mapping["n"] == 203
    # the low-confidence items are not mapped to a confidence of its own block
    low = apply_confidence_map(mapping, [0.05])[0]
    assert low > 0.0, low
    assert sizes == sorted(sizes)


def test_apply_clamps_outside_the_fitted_range():
    confidences = [0.5, 0.6, 0.7]
    correct = [False, True, True]
    mapping = fit_confidence_map(confidences, correct, alpha=0.0, min_n=1)
    below = apply_confidence_map(mapping, [0.01])[0]
    above = apply_confidence_map(mapping, [0.99])[0]
    assert 0.0 <= below <= 1.0
    assert 0.0 <= above <= 1.0


def test_report_improves_ece_on_an_overconfident_holdout():
    fit_conf, fit_ok = overconfident(3000, seed=1)
    hold_conf, hold_ok = overconfident(2000, seed=2)
    report = confidence_map_report(fit_conf, fit_ok, hold_conf, hold_ok)
    assert report["n_fit"] == 3000
    assert report["n_holdout"] == 2000
    assert report["after"] < report["before"], report
    assert report["after"] < 0.06, report


def test_report_is_deterministic():
    fit_conf, fit_ok = overconfident(500, seed=3)
    hold_conf, hold_ok = overconfident(500, seed=4)
    first = confidence_map_report(fit_conf, fit_ok, hold_conf, hold_ok)
    second = confidence_map_report(fit_conf, fit_ok, hold_conf, hold_ok)
    assert first == second
