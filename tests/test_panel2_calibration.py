"""Tests for the per-type temperature and slot-bias fit (#338)."""

from __future__ import annotations

import math
import random

import pytest

from model.evaluation.calibration import (
    fit_temperature,
    fit_temperature_bias,
    multiclass_brier,
    softmax_logits,
    top1_calibration,
)


def sample_labels(logits, temperature=1.0, bias=None, rng=None):
    """Draw one label per row from the model's own distribution."""
    labels = []
    for row in logits:
        probabilities = softmax_logits(row, temperature, bias)
        threshold = rng.random()
        total = 0.0
        chosen = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            total += probability
            if threshold <= total:
                chosen = index
                break
        labels.append(chosen)
    return labels


def random_logits(rows, width, seed, scale=1.0):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, scale) for _ in range(width)] for _ in range(rows)]


def test_temperature_cannot_move_the_argmax():
    logits = random_logits(200, 5, seed=0, scale=3.0)
    baseline = [max(range(len(row)), key=row.__getitem__) for row in logits]
    for temperature in (0.05, 0.3, 1.0, 2.5, 20.0):
        moved = [
            max(range(len(row)), key=row.__getitem__)
            for row in (softmax_logits(row, temperature) for row in logits)
        ]
        assert moved == baseline
        for row in logits:
            probabilities = softmax_logits(row, temperature)
            assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12)


def test_uniform_bias_cancels_in_the_softmax():
    logits = random_logits(50, 4, seed=1, scale=2.0)
    for row in logits:
        plain = softmax_logits(row, 1.7)
        shifted = softmax_logits(row, 1.7, [0.9, 0.9, 0.9, 0.9])
        assert plain == pytest.approx(shifted, rel=1e-12)


def test_fit_temperature_recovers_a_planted_scale():
    rng = random.Random(7)
    logits = random_logits(3000, 4, seed=2, scale=2.0)
    labels = sample_labels(logits, temperature=2.0, rng=rng)
    fitted = fit_temperature(logits, labels, criterion="nll")
    assert fitted["temperature"] == pytest.approx(2.0, rel=0.15)
    assert fitted["nll_after"] <= fitted["nll_before"]


def test_fit_temperature_bias_recovers_a_planted_slot_prior():
    rng = random.Random(11)
    planted = [0.0, 1.2, -0.8, 0.4]
    logits = random_logits(600, 4, seed=3, scale=1.0)
    labels = sample_labels(logits, temperature=1.0, bias=planted, rng=rng)
    fitted = fit_temperature_bias(logits, labels, sweeps=8)
    # The softmax is invariant to an additive constant on the bias, so only the
    # centred vector is identifiable; the ridge pins the mean near zero.
    centred = [value - sum(fitted["bias"]) / len(fitted["bias"]) for value in fitted["bias"]]
    planted_centred = [value - sum(planted) / len(planted) for value in planted]
    for got, want in zip(centred, planted_centred):
        assert got == pytest.approx(want, abs=0.35)
    assert fitted["nll_after"] <= fitted["nll_before"]


def test_slot_bias_can_move_accuracy_and_temperature_cannot():
    # Two options; the raw logits favour option 1 but the gold is option 0 in
    # most rows, so a slot-0 bias is the only parameter in this family that can
    # flip the answer.
    logits = [[0.0, 0.6] for _ in range(100)]
    labels = [0] * 90 + [1] * 10
    temperature = fit_temperature(logits, labels, criterion="nll")
    unchanged = [
        max(range(2), key=p.__getitem__)
        for p in (softmax_logits(row, temperature["temperature"]) for row in logits)
    ]
    assert unchanged == [1] * 100
    joint = fit_temperature_bias(logits, labels, sweeps=10)
    flipped = [
        max(range(2), key=p.__getitem__)
        for p in (softmax_logits(row, joint["temperature"], joint["bias"]) for row in logits)
    ]
    assert flipped == [0] * 100


def test_brier_penalises_mass_on_the_wrong_option():
    one_hot = [[0.0, 1.0]]
    flat = [[0.5, 0.5]]
    assert multiclass_brier(one_hot, [1])["brier"] == 0.0
    assert multiclass_brier(flat, [1])["brier"] == pytest.approx(0.5)
    assert multiclass_brier(one_hot, [0])["brier"] == pytest.approx(2.0)


def test_top1_calibration_reports_ece_over_confidence():
    # Ten items at confidence 0.8, eight correct: calibration gap is 0.0.
    probabilities = [[0.8, 0.2] for _ in range(10)]
    labels = [0] * 8 + [1] * 2
    assert top1_calibration(probabilities, labels, 10)["ece"] == pytest.approx(0.0)
    # Five correct out of ten at 0.8 confidence: gap 0.3 in one bin.
    labels = [0] * 5 + [1] * 5
    assert top1_calibration(probabilities, labels, 10)["ece"] == pytest.approx(0.3)
