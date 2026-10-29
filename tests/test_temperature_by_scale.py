"""Tests for the per-scale temperature comparison.

Torch-free on purpose: CI installs only pytest, so the logic that decides
whether a per-scale temperature beats a shared one has to live in
``model/evaluation/calibration.py`` rather than in the eval script.
"""

from __future__ import annotations

import random

import pytest

from model.evaluation.calibration import temperature_by_scale


def _group(scale: int, n: int, sharpness: float, rng: random.Random):
    """Logits with a fixed signal, sharpened by ``sharpness``.

    Multiplying a logit row by ``s`` is the same as dividing its temperature by
    ``s``, so a group built with ``s = 3`` needs a temperature three times larger
    than one built with ``s = 1`` to reach the same calibration.
    """
    logits, labels = [], []
    for _ in range(n):
        label = rng.randrange(scale)
        row = [rng.gauss(0.0, 1.0) for _ in range(scale)]
        row[label] += 2.0
        logits.append([value * sharpness for value in row])
        labels.append(label)
    return logits, labels


def _two_scale_set(n: int = 400, seed: int = 7):
    rng = random.Random(seed)
    logits_5, labels_5 = _group(5, n, 1.0, rng)
    logits_6, labels_6 = _group(6, n, 3.0, rng)
    return logits_5 + logits_6, labels_5 + labels_6, [5] * n + [6] * n


def test_per_scale_beats_a_shared_temperature_on_the_same_items():
    logits, labels, scales = _two_scale_set()
    report = temperature_by_scale(logits, labels, scales, steps=120, seed=3)

    pooled = report["pooled"]
    # The pooled figures are over the holdout halves, which is what makes the
    # shared-vs-per-scale comparison matched: identical items, identical split.
    assert pooled["n"] == sum(e["n_holdout"] for e in report["by_scale"].values())
    assert pooled["n"] < len(logits)
    assert pooled["ece_per_scale"] < pooled["ece_shared"]


def test_the_two_scales_get_different_temperatures():
    """The sharpened group should need the larger temperature."""
    logits, labels, scales = _two_scale_set()
    report = temperature_by_scale(logits, labels, scales, steps=120, seed=3)

    by_scale = report["by_scale"]
    assert by_scale["5"]["temperature"] < by_scale["6"]["temperature"]
    assert not by_scale["5"]["borrowed"]
    assert not by_scale["6"]["borrowed"]


def test_every_scale_reports_before_and_after_on_its_own_holdout():
    logits, labels, scales = _two_scale_set()
    report = temperature_by_scale(logits, labels, scales, steps=120, seed=3)

    for entry in report["by_scale"].values():
        assert entry["n"] == entry["n_fit"] + entry["n_holdout"]
        assert entry["ece_own_t"] <= entry["ece_at_t1"] + 1e-9


def test_a_small_scale_borrows_the_shared_temperature():
    logits, labels, scales = _two_scale_set(n=200)
    # A three-item scale cannot support its own fit; it must be marked, not
    # silently fitted on three items.
    logits += [[0.1, -0.2, 0.0], [0.3, 0.0, -0.1], [0.0, 0.2, -0.3]]
    labels += [0, 2, 1]
    scales += [3, 3, 3]

    report = temperature_by_scale(logits, labels, scales, steps=60, seed=3)
    small = report["by_scale"]["3"]
    assert small["borrowed"] is True
    assert "fit" not in small
    assert small["temperature"] == pytest.approx(report["shared"]["temperature"])


def test_the_report_is_deterministic():
    logits, labels, scales = _two_scale_set(n=120)
    first = temperature_by_scale(logits, labels, scales, steps=60, seed=11)
    second = temperature_by_scale(logits, labels, scales, steps=60, seed=11)
    assert first == second


def test_an_empty_input_returns_an_empty_report():
    report = temperature_by_scale([], [], [])
    assert report["n_items"] == 0
    assert report["by_scale"] == {}
    assert report["pooled"] == {}


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        temperature_by_scale([[0.0, 1.0]], [1, 0], [2])
