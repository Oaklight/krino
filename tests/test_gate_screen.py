"""Tests for the three-state gate screen (#263, research/37-gate-policy.md).

Each synthetic case is specified by its raw paired accuracy difference, the
scale the power analysis in #263 uses. The items are three-option items, so
their chance level is 1/3 and a raw difference of d pp is about 1.5d pp of
chance-corrected accuracy; JevBench's mean per-item chance is 0.318 (factor
1.47), close enough that the cases read at JevBench's scale. The states are
assigned on the chance-corrected interval against the 12 pp cc margin.
"""

from __future__ import annotations

import pytest

from model.evaluation.gate import (
    COARSE_SAFETY_SCREEN_PASSED,
    IMPROVEMENT_SUGGESTED_BUT_UNRESOLVED,
    IMPROVEMENT_SUPPORTED,
    JEVBENCH_MARGIN_PP,
    MATERIAL_REGRESSION_SUPPORTED,
    UNRESOLVED,
    PairedOutcome,
    paired_screen,
)

JEVBENCH_ITEMS = 231


def paired(
    n: int,
    wins: int,
    losses: int,
    n_options: int | tuple[int, ...] = 3,
) -> tuple[dict[str, PairedOutcome], dict[str, PairedOutcome]]:
    """Arm/control outcomes with ``wins`` arm-only and ``losses`` control-only.

    Every remaining item is correct under both readings, so it adds nothing to
    the discordant counts or the paired difference. ``n_options`` is either one
    count for every item or a per-item tuple.
    """
    options = (n_options,) * n if isinstance(n_options, int) else n_options
    assert len(options) == n
    arm: dict[str, PairedOutcome] = {}
    control: dict[str, PairedOutcome] = {}
    for index in range(n):
        item_id = f"item{index:03d}"
        options_here = options[index]
        if index < wins:
            arm_correct, control_correct = True, False
        elif index < wins + losses:
            arm_correct, control_correct = False, True
        else:
            arm_correct, control_correct = True, True
        arm[item_id] = PairedOutcome(arm_correct, options_here)
        control[item_id] = PairedOutcome(control_correct, options_here)
    return arm, control


def test_jevbench_margin_is_twelve_points_of_chance_corrected_accuracy():
    assert JEVBENCH_MARGIN_PP == 12.0


def test_perfect_replication_passes_the_screen_and_claims_no_improvement():
    arm, control = paired(JEVBENCH_ITEMS, 0, 0)
    screen = paired_screen(arm, control)

    assert screen.states == {COARSE_SAFETY_SCREEN_PASSED}
    assert screen.action == "no objection from the coarse screen"
    assert screen.delta_pp == 0.0
    assert screen.ci_low_pp == 0.0
    assert screen.ci_high_pp == 0.0
    assert screen.p_value == 1.0
    assert screen.detectable_pp is None


def test_six_point_deficit_is_unresolved_not_a_material_regression():
    # 20 wins / 34 losses is a raw deficit of 6.1 pp, 9.1 pp chance-corrected
    # at three options. The interval crosses -12 pp, so this is unresolved.
    arm, control = paired(JEVBENCH_ITEMS, 20, 34)
    screen = paired_screen(arm, control)

    assert screen.ci_low_pp < -12.0 < screen.ci_high_pp
    assert screen.states == {UNRESOLVED}
    assert screen.action == "investigate"
    assert MATERIAL_REGRESSION_SUPPORTED not in screen.states


def test_a_point_deficit_below_the_margin_can_still_be_unresolved():
    # 17 wins / 38 losses is 13.6 pp chance-corrected, below the -12 pp
    # margin, and the interval still crosses it. A point rule calls this a
    # material regression; the interval says the item count cannot resolve it.
    arm, control = paired(JEVBENCH_ITEMS, 17, 38)
    screen = paired_screen(arm, control)

    assert screen.delta_pp < -JEVBENCH_MARGIN_PP
    assert screen.ci_low_pp < -JEVBENCH_MARGIN_PP < screen.ci_high_pp
    assert screen.states == {UNRESOLVED}


def test_eighteen_point_deficit_is_a_material_regression():
    arm, control = paired(JEVBENCH_ITEMS, 6, 48)
    screen = paired_screen(arm, control)

    assert screen.ci_high_pp < -12.0
    assert screen.states == {MATERIAL_REGRESSION_SUPPORTED}
    assert screen.action == "block promotion"


def test_six_point_improvement_is_suggested_but_unresolved():
    arm, control = paired(JEVBENCH_ITEMS, 34, 20)
    screen = paired_screen(arm, control)

    assert screen.delta_pp > 0.0
    assert screen.ci_low_pp < 0.0 < screen.ci_high_pp
    assert screen.states == {
        COARSE_SAFETY_SCREEN_PASSED,
        IMPROVEMENT_SUGGESTED_BUT_UNRESOLVED,
    }
    assert IMPROVEMENT_SUPPORTED not in screen.states


def test_twelve_point_improvement_is_supported():
    arm, control = paired(JEVBENCH_ITEMS, 41, 13)
    screen = paired_screen(arm, control)

    assert screen.ci_low_pp > 0.0
    assert screen.states == {COARSE_SAFETY_SCREEN_PASSED, IMPROVEMENT_SUPPORTED}


def test_detectable_difference_uses_the_measured_discordance():
    arm, control = paired(JEVBENCH_ITEMS, 6, 48)
    screen = paired_screen(arm, control)

    assert screen.n == JEVBENCH_ITEMS
    assert screen.discordance == pytest.approx(54 / JEVBENCH_ITEMS)
    assert screen.detectable_pp == pytest.approx(9.0, abs=0.05)
    # Raw to chance-corrected at a mean chance of 1/3.
    assert screen.detectable_cc_pp == pytest.approx(9.0 / (1 - 1 / 3), abs=0.08)


def test_interval_matches_a_hand_computed_case():
    # Seven wins, one loss, eight items, two options: d = +2 (x7), -2 (x1).
    # mean 1.5, sd sqrt(2), se 0.5, 95% half-width 0.97998.
    arm, control = paired(8, 7, 1, n_options=2)
    screen = paired_screen(arm, control)

    assert screen.delta_pp == pytest.approx(150.0)
    assert screen.ci_low_pp == pytest.approx(52.0, abs=0.01)
    assert screen.ci_high_pp == pytest.approx(248.0, abs=0.01)
    # Exact two-sided McNemar: 2 * (C(8,0) + C(8,1)) * 0.5**8.
    assert screen.p_value == pytest.approx(18 / 256)


def test_each_item_uses_its_own_chance_level():
    arm = {
        "a": PairedOutcome(True, 2),
        "b": PairedOutcome(True, 4),
        "c": PairedOutcome(False, 2),
        "d": PairedOutcome(True, 4),
    }
    control = {
        "a": PairedOutcome(False, 2),
        "b": PairedOutcome(False, 4),
        "c": PairedOutcome(True, 2),
        "d": PairedOutcome(True, 4),
    }
    # Signed differences: +2, +4/3, -2, 0.
    screen = paired_screen(arm, control)

    assert screen.delta_pp == pytest.approx((2 + 4 / 3 - 2) / 4 * 100)


def test_exact_p_agrees_with_the_mcnemar_module():
    from model.scripts.paired_mcnemar import mcnemar

    arm, control = paired(20, 12, 5)
    ids = sorted(arm)
    screen = paired_screen(arm, control, ids)
    first = {item_id: arm[item_id].correct for item_id in ids}
    second = {item_id: control[item_id].correct for item_id in ids}

    assert (screen.wins, screen.losses, screen.p_value) == mcnemar(
        first, second, ids
    )


def test_screen_rejects_a_single_shared_item():
    arm, control = paired(1, 0, 1)
    with pytest.raises(ValueError):
        paired_screen(arm, control)


def test_screen_rejects_a_missing_id():
    arm, control = paired(4, 1, 1)
    with pytest.raises(KeyError):
        paired_screen(arm, control, ["item000", "item001", "absent"])


def test_screen_rejects_mismatched_option_counts():
    arm, control = paired(4, 1, 1)
    control["item000"] = PairedOutcome(control["item000"].correct, 4)
    with pytest.raises(ValueError):
        paired_screen(arm, control)


def test_screen_rejects_an_invalid_alpha():
    arm, control = paired(4, 1, 1)
    for alpha in (0.0, 1.0, -0.05):
        with pytest.raises(ValueError):
            paired_screen(arm, control, alpha=alpha)


def test_a_paired_outcome_needs_at_least_two_options():
    with pytest.raises(ValueError):
        PairedOutcome(True, 1)
