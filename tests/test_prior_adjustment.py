"""Tests for the class-prior correction and the QWK aggregation it feeds.

Pure Python, so it runs without torch (and in CI on a bare interpreter), and
it lives in ``model/evaluation`` for the same reason ``calibration`` does: the
arithmetic that decides a research claim should be checkable on its own.

The invariants pinned here are the ones a wrong implementation would quietly
break: alpha=0 must be the identity, a full correction must actually move the
predicted marginal onto the reference, ties must not depend on list position,
and the aggregate must drop scales too small to read rather than let them move
the mean.
"""

from __future__ import annotations

import math

import pytest

from model.evaluation.prior_adjustment import (
    MIN_SCALE_ITEMS,
    adjusted_argmax,
    log_prior_shift,
    marginal,
    qwk_by_scale,
)


class TestMarginal:
    def test_counts_normalise_to_one(self):
        assert marginal([0, 1, 2, 3], 4) == [0.25] * 4

    def test_unseen_levels_are_zero_and_kept(self):
        # A level absent from the batch must still occupy a slot: dropping it
        # would silently shorten the vector the shift is added to.
        assert marginal([0, 0, 2], 3) == [pytest.approx(2 / 3), 0.0, pytest.approx(1 / 3)]

    def test_empty_batch_returns_zeros_not_nan(self):
        assert marginal([], 4) == [0.0] * 4

    def test_out_of_range_index_is_an_error(self):
        with pytest.raises(ValueError):
            marginal([0, 4], 4)
        with pytest.raises(ValueError):
            marginal([-1], 4)


class TestLogPriorShift:
    def test_alpha_zero_is_the_identity(self):
        assert log_prior_shift([0.2, 0.8], [0.5, 0.5], 0.0) == [0.0, 0.0]

    def test_refuses_mismatched_lengths(self):
        with pytest.raises(ValueError):
            log_prior_shift([1.0], [1.0, 1.0], 1.0)

    def test_never_returns_an_infinite_shift(self):
        # A reference that assigns a level zero probability would make log(0)
        # infinite and turn the argmax into a coin flip between NaNs.
        shift = log_prior_shift([0.5, 0.5], [1.0, 0.0], 1.0)
        assert all(math.isfinite(value) for value in shift)

    def test_raises_the_under_predicted_level(self):
        # Level 0 is under-predicted (0.1 against a reference 0.4), so its shift
        # must be positive relative to level 1's.
        shift = log_prior_shift([0.1, 0.9], [0.4, 0.6], 1.0)
        assert shift[0] > shift[1]


class TestAdjustedArgmax:
    def test_alpha_zero_matches_the_plain_argmax(self):
        logits = [0.1, 2.0, 0.5]
        assert adjusted_argmax(logits, [0.3, 0.4, 0.3], [0.1, 0.1, 0.8], 0.0) == 1

    def test_full_correction_flips_an_under_predicted_level(self):
        # The head narrowly prefers level 1 but the reference says level 0 is
        # common and level 1 rare; a full correction should move the answer.
        logits = [0.0, 0.5, -1.0]
        observed = [0.05, 0.85, 0.10]
        reference = [0.60, 0.20, 0.20]
        assert adjusted_argmax(logits, observed, reference, 0.0) == 1
        assert adjusted_argmax(logits, observed, reference, 1.0) == 0

    def test_shift_is_scale_invariant_in_the_logits(self):
        # Adding a constant to every logit changes nothing, so a correction that
        # depends on the absolute scale would be measuring the wrong thing.
        observed, reference = [0.2, 0.8], [0.5, 0.5]
        first = adjusted_argmax([0.0, 1.0], observed, reference, 1.0)
        second = adjusted_argmax([100.0, 101.0], observed, reference, 1.0)
        assert first == second

    def test_ties_break_toward_the_lower_level_regardless_of_input_order(self):
        # Levels are always in canonical order here, but a tie must still be
        # decided by level, never by where the level sat in the list.
        assert adjusted_argmax([1.0, 1.0, 1.0], [1 / 3] * 3, [1 / 3] * 3, 1.0) == 0

    def test_unguarded_readme_example(self):
        # The STS-B case from research/18, in miniature: the head follows the
        # training marginal, the split is dominated by level 0.
        logits = [1.0, 2.5, 2.4, 2.6, 2.2, 1.5]
        observed = [0.14, 0.15, 0.14, 0.23, 0.26, 0.08]
        reference = [0.53, 0.11, 0.08, 0.12, 0.12, 0.03]
        assert adjusted_argmax(logits, observed, reference, 0.0) == 3
        assert adjusted_argmax(logits, observed, reference, 1.0) == 0


class TestQwkByScale:
    def test_perfect_predictions_score_one_per_scale(self):
        result = qwk_by_scale({4: [0, 1, 2, 3] * 10}, {4: [0, 1, 2, 3] * 10})
        assert result["by_scale"]["4"]["qwk"] == pytest.approx(1.0)
        assert result["mixed_scale_qwk"] == pytest.approx(1.0)

    def test_mixed_mean_weights_scales_by_item_count(self):
        # Two readable scales: perfect on the small one, chance-level on the big
        # one. An unweighted mean would report the small scale's 1.0 twice.
        gold = {4: [0, 1, 2, 3] * 10, 5: [0, 1, 2, 3, 4] * 20}
        predictions = {4: [0, 1, 2, 3] * 10, 5: [2] * 100}
        result = qwk_by_scale(predictions, gold)
        expected = (
            result["by_scale"]["4"]["qwk"] * 40 + result["by_scale"]["5"]["qwk"] * 100
        ) / 140
        assert result["mixed_scale_qwk"] == pytest.approx(expected)

    def test_scales_below_the_floor_are_excluded_and_named(self):
        gold = {3: [0, 1, 2], 4: [0, 1, 2, 3] * 10}
        predictions = {3: [0, 1, 2], 4: [0, 1, 2, 3] * 10}
        result = qwk_by_scale(predictions, gold)
        assert MIN_SCALE_ITEMS > 3
        assert result["excluded_scales"] == ["3"]
        assert result["mixed_scale_qwk"] == pytest.approx(result["by_scale"]["4"]["qwk"])

    def test_all_scales_excluded_reports_zero_not_a_division_error(self):
        result = qwk_by_scale({3: [0, 1, 2]}, {3: [0, 1, 2]})
        assert result["mixed_scale_qwk"] == 0.0
        assert result["mixed_scale_accuracy"] == 0.0
        assert result["excluded_scales"] == ["3"]

    def test_mismatched_scales_are_an_error(self):
        with pytest.raises(ValueError):
            qwk_by_scale({4: [0, 1, 2, 3]}, {5: [0, 1, 2, 3, 4]})

    def test_mismatched_item_counts_are_an_error(self):
        with pytest.raises(ValueError):
            qwk_by_scale({4: [0, 1, 2, 3]}, {4: [0, 1, 2]})
