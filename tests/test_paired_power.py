"""Tests for the exact paired-McNemar power helpers."""

from __future__ import annotations

import pytest

from model.evaluation.paired_power import (
    detectable_delta,
    items_for_power,
    power,
    two_sided_p,
)


def test_two_sided_p_is_symmetric():
    assert two_sided_p(10, 3) == pytest.approx(two_sided_p(3, 10))


def test_two_sided_p_with_no_discordant_pairs_is_one():
    assert two_sided_p(0, 0) == 1.0


def test_two_sided_p_matches_hand_computation():
    # 12 discordant items, 3 in the smaller direction.
    expected = 2 * sum(
        __import__("math").comb(12, index) for index in range(4)
    ) * (0.5**12)
    assert two_sided_p(9, 3) == pytest.approx(expected)


def test_two_sided_p_rejects_negative_counts():
    with pytest.raises(ValueError):
        two_sided_p(-1, 3)


def test_power_at_zero_effect_is_at_most_alpha():
    # Under the null the exact test is conservative, never anti-conservative.
    assert power(300, 0.324, 0.0) <= 0.05 + 1e-12


def test_power_grows_with_the_effect():
    small = power(111, 0.324, 0.05)
    large = power(111, 0.324, 0.20)
    assert small < large


def test_power_grows_with_item_count():
    small = power(111, 0.324, 0.10)
    large = power(500, 0.324, 0.10)
    assert small < large


def test_power_is_symmetric_in_the_sign_of_the_effect():
    assert power(111, 0.324, 0.12) == pytest.approx(power(111, 0.324, -0.12))


def test_power_matches_the_measured_hard_tier():
    # 111 items, 36 discordant, a ten-point gain: about 38% power.
    assert power(111, 36 / 111, 0.10) == pytest.approx(0.376, abs=0.01)
    # The same set reaches the conventional 80% only near sixteen points.
    assert power(111, 36 / 111, 0.15) == pytest.approx(0.758, abs=0.01)


def test_power_rejects_bad_arguments():
    with pytest.raises(ValueError):
        power(0, 0.3, 0.1)
    with pytest.raises(ValueError):
        power(100, 0.0, 0.1)
    with pytest.raises(ValueError):
        power(100, 1.5, 0.1)
    with pytest.raises(ValueError):
        power(100, 0.3, 0.1, alpha=1.0)


def test_items_for_power_finds_the_boundary():
    needed = items_for_power(0.324, 0.10)
    assert needed is not None
    assert 250 <= needed <= 320
    # One item fewer must fall short and the answer itself must clear the bar.
    assert power(needed, 0.324, 0.10) >= 0.8
    assert power(needed - 1, 0.324, 0.10) < 0.8


def test_items_for_power_at_ninety_percent_costs_more_items():
    eighty = items_for_power(0.324, 0.10, target=0.8)
    ninety = items_for_power(0.324, 0.10, target=0.9)
    assert eighty is not None and ninety is not None
    assert ninety > eighty


def test_items_for_power_rejects_bad_target():
    with pytest.raises(ValueError):
        items_for_power(0.3, 0.1, target=1.0)
    with pytest.raises(ValueError):
        items_for_power(0.3, 0.0)


def test_detectable_delta_on_the_hard_tier():
    # 111 items at 32% discordance: about sixteen points at 80% power.
    assert detectable_delta(111, 0.324) == pytest.approx(0.16, abs=0.01)


def test_detectable_delta_falls_with_more_items():
    assert detectable_delta(600, 0.324) < detectable_delta(111, 0.324)


def test_detectable_delta_rejects_bad_target():
    with pytest.raises(ValueError):
        detectable_delta(111, 0.324, target=0.0)
