"""Tests for the ordinal decoders.

The point of these functions is that a QWK-scored task wants the posterior mean
rather than the mode, so the tests pin the cases where the two disagree: a
distribution whose mass sits between two levels decides differently under each
rule, and the change must be a consequence of the cost structure rather than a
tuning knob. Pure Python, no torch.
"""

from __future__ import annotations

import math

import pytest

from model.evaluation.ordinal_decode import (
    DECODERS,
    decode_argmax,
    decode_expected_level,
    decode_ordinal_median,
)


def _logits(*probabilities: float) -> list[float]:
    """Logits that softmax to the given probabilities."""
    return [math.log(p) for p in probabilities]


class TestArgmax:
    def test_picks_the_most_probable_level(self):
        assert decode_argmax(_logits(0.1, 0.7, 0.2)) == 1

    def test_ties_break_toward_the_lower_level(self):
        assert decode_argmax(_logits(0.4, 0.4)) == 0

    def test_ignores_order_of_magnitude(self):
        assert decode_argmax([1.0, 2.0, 3.0]) == 2

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            decode_argmax([])


class TestOrdinalMedian:
    def test_returns_the_level_at_half_the_mass(self):
        # 0.45 on level 0, then 0.35 on level 1 reaches 0.80 > 0.5 at level 1.
        assert decode_ordinal_median(_logits(0.45, 0.35, 0.20)) == 1

    def test_is_the_smallest_level_reaching_a_half(self):
        assert decode_ordinal_median(_logits(0.6, 0.3, 0.1)) == 0

    def test_a_confident_tail_still_lands_low(self):
        assert decode_ordinal_median(_logits(0.55, 0.10, 0.10, 0.25)) == 0

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            decode_ordinal_median([])


class TestExpectedLevel:
    def test_is_the_posterior_mean_rounded(self):
        # mean = 0*0.2 + 1*0.5 + 2*0.3 = 1.1 -> 1
        assert decode_expected_level(_logits(0.2, 0.5, 0.3)) == 1

    def test_rounds_a_mean_above_the_halfway_point(self):
        # mean = 0*0.3 + 1*0.7 = 0.7 -> 1
        assert decode_expected_level(_logits(0.3, 0.7)) == 1

    def test_a_half_mean_rounds_to_the_even_level(self):
        # Python's round() is half-to-even, so an exact 0.5 goes to 0. The case
        # is measure-zero on real distributions; pinning it stops a later
        # "fix" to half-up from silently changing every reported number.
        assert decode_expected_level(_logits(0.5, 0.5)) == 0

    def test_clamps_to_the_scale(self):
        # A negative bias cannot push below level 0.
        assert decode_expected_level(_logits(0.99, 0.01), bias=-5.0) == 0

    def test_bias_shifts_the_choice(self):
        base = decode_expected_level(_logits(0.4, 0.6))
        assert decode_expected_level(_logits(0.4, 0.6), bias=-1.0) < base

    def test_agrees_with_argmax_on_a_point_mass(self):
        assert decode_expected_level(_logits(0.001, 0.001, 0.998)) == 2

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            decode_expected_level([])


class TestDisagreement:
    def test_the_decoders_differ_exactly_where_the_cost_structure_bites(self):
        # A bimodal distribution: the mode is level 0, but the mean is 1.0, so
        # the squared-error rule takes level 1. This is the mechanism the note
        # reports -- it is a property of the decision rule, not a fitted knob.
        logits = _logits(0.45, 0.10, 0.45)
        assert decode_argmax(logits) == 0
        assert decode_expected_level(logits) == 1

    def test_registry_exposes_all_three(self):
        assert set(DECODERS) == {"argmax", "ordinal_median", "expected_level"}
