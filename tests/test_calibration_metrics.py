"""Tests for the multiclass and ordinal calibration metrics.

Pure Python, so it runs without torch (and in CI on a bare interpreter).
"""

from __future__ import annotations

import math

import pytest

from model.evaluation.calibration import (
    fit_temperature,
    multiclass_brier,
    quadratic_weighted_kappa,
    softmax,
    top1_calibration,
)


class TestMulticlassBrier:
    def test_perfect_prediction_scores_zero(self):
        result = multiclass_brier([[1.0, 0.0], [0.0, 1.0]], [0, 1])
        assert result["brier"] == pytest.approx(0.0)
        assert result["n"] == 2

    def test_known_value(self):
        # Item 1: 0.09 + 0.04 + 0.01 = 0.14; item 2: 0.04 + 0.25 + 0.09 = 0.38.
        result = multiclass_brier([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], [0, 1])
        assert result["brier"] == pytest.approx((0.14 + 0.38) / 2)

    def test_penalises_wrong_option_mass(self):
        """A confident wrong distribution must cost more than a flat one."""
        confident = multiclass_brier([[0.02, 0.96, 0.02]], [0])["brier"]
        flat = multiclass_brier([[1 / 3, 1 / 3, 1 / 3]], [0])["brier"]
        assert confident > flat

    def test_empty(self):
        assert multiclass_brier([], [])["n"] == 0


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self):
        assert quadratic_weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2], 3) == pytest.approx(1.0)

    def test_off_by_one_costs_less_than_off_by_two(self):
        labels = [0, 1, 2, 3]
        near = quadratic_weighted_kappa([1, 2, 3, 3], labels, 4)
        far = quadratic_weighted_kappa([2, 3, 3, 3], labels, 4)
        assert near > far

    def test_known_off_by_one_value(self):
        # predictions [1,2,3] vs labels [0,1,2] on 4 levels: kappa = 4/7.
        assert quadratic_weighted_kappa([1, 2, 3], [0, 1, 2], 4) == pytest.approx(4 / 7)

    def test_chance_level_for_symmetric_disagreement(self):
        assert quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 0, 1], 2) == pytest.approx(0.0)

    def test_constant_labels_have_no_variance(self):
        assert quadratic_weighted_kappa([1, 1], [1, 1], 3) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([0, 1], [0], 3)

    def test_single_level_is_degenerate(self):
        assert quadratic_weighted_kappa([0], [0], 1) == 0.0


class TestSoftmax:
    def test_sums_to_one(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert sum(probs) == pytest.approx(1.0)

    def test_stable_for_large_logits(self):
        probs = softmax([1000.0, 0.0])
        assert probs[0] == pytest.approx(1.0)
        assert not math.isnan(probs[1])

    def test_empty(self):
        assert softmax([]) == []


class TestFitTemperature:
    def test_overconfident_and_wrong_needs_temperature_above_one(self):
        """100 logits of magnitude 6 with 25% of them wrong must be softened."""
        logits = [[6.0, 0.0]] * 75 + [[6.0, 0.0]] * 25
        labels = [0] * 75 + [1] * 25
        result = fit_temperature(logits, labels)
        assert result["temperature"] > 1.0
        assert result["nll_after"] < result["nll_before"]

    def test_calibrated_logits_keep_temperature_near_one(self):
        logits = [[0.7, -0.7]] * 100
        labels = [0] * 69 + [1] * 31
        result = fit_temperature(logits, labels)
        assert 0.5 < result["temperature"] < 2.0

    def test_reduces_ece(self):
        logits = [[4.0, 0.0]] * 100
        labels = [0] * 70 + [1] * 30
        fitted = fit_temperature(logits, labels)
        scaled = [[value / fitted["temperature"] for value in row] for row in logits]
        before = top1_calibration([softmax(row) for row in logits], labels)["ece"]
        after = top1_calibration([softmax(row) for row in scaled], labels)["ece"]
        assert after < before

    def test_empty(self):
        result = fit_temperature([], [])
        assert result["temperature"] == 1.0
        assert result["n"] == 0


class TestTop1Calibration:
    def test_confident_and_correct_has_low_ece(self):
        probs = [[0.95, 0.05]] * 100
        labels = [0] * 95 + [1] * 5
        assert top1_calibration(probs, labels)["ece"] < 0.1

    def test_confident_and_wrong_has_high_ece(self):
        probs = [[0.95, 0.05]] * 100
        labels = [1] * 100
        assert top1_calibration(probs, labels)["ece"] > 0.8


class TestTemperatureCurve:
    """NLL and ECE need not agree on the best temperature."""

    @staticmethod
    def _overconfident():
        # Every item is predicted at 0.9 confidence and right 60% of the time.
        logits = [[3.0, 0.0]] * 10 + [[0.0, 3.0]] * 10
        labels = [0] * 6 + [1] * 4 + [1] * 6 + [0] * 4
        return logits, labels

    def test_reports_both_criteria_and_t_one(self):
        from model.evaluation.calibration import temperature_curve

        logits, labels = self._overconfident()
        result = temperature_curve(logits, labels)
        assert result["nll_best"]["temperature"] > 1.0
        assert result["at_one"]["temperature"] == 1.0
        for key in ("temperature", "nll", "ece"):
            assert key in result["nll_best"]
        assert len(result["curve"]) == 41  # 40 log-spaced points plus T=1

    def test_the_two_minima_can_differ(self):
        from model.evaluation.calibration import temperature_curve

        logits, labels = self._overconfident()
        result = temperature_curve(logits, labels)
        assert result["ece_best"]["ece"] <= result["at_one"]["ece"]
        assert result["nll_best"]["nll"] <= result["at_one"]["nll"]

    def test_empty_input(self):
        from model.evaluation.calibration import temperature_curve

        assert temperature_curve([], [])["curve"] == []


class TestFitTemperatureCriterion:
    """The fitting criterion is a choice, and on score it changes the answer."""

    @staticmethod
    def _skewed():
        # Right 60% of the time at 0.95 confidence: overconfident and skewed.
        rows = []
        for index in range(60):
            rows.append(([4.0, 0.0, 0.0], 0))
        for index in range(20):
            rows.append(([4.0, 0.0, 0.0], 1))
        for index in range(20):
            rows.append(([0.0, 4.0, 0.0], 1))
        return [row for row, _ in rows], [label for _, label in rows]

    def test_default_is_nll(self):
        from model.evaluation.calibration import fit_temperature

        logits, labels = self._skewed()
        assert fit_temperature(logits, labels)["criterion"] == "nll"

    def test_ece_criterion_returns_a_temperature_too(self):
        from model.evaluation.calibration import fit_temperature

        logits, labels = self._skewed()
        fitted = fit_temperature(logits, labels, criterion="ece")
        assert fitted["temperature"] > 0
        assert fitted["criterion"] == "ece"

    def test_the_ece_criterion_does_not_lose_on_its_own_metric(self):
        """Whatever else changes, fitting for ECE cannot be worse at ECE."""
        from model.evaluation.calibration import fit_temperature, softmax, top1_calibration

        logits, labels = self._skewed()

        def ece_at(temperature):
            probabilities = [
                softmax([value / temperature for value in row]) for row in logits
            ]
            return top1_calibration(probabilities, labels)["ece"]

        by_nll = fit_temperature(logits, labels, criterion="nll")["temperature"]
        by_ece = fit_temperature(logits, labels, criterion="ece")["temperature"]
        assert ece_at(by_ece) <= ece_at(by_nll) + 1e-9

    def test_unknown_criterion_raises(self):
        import pytest
        from model.evaluation.calibration import fit_temperature

        with pytest.raises(ValueError):
            fit_temperature([[1.0, 0.0]], [0], criterion="brier")
