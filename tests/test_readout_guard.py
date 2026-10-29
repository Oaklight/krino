"""Tests for the long-schedule/readout guard."""

from __future__ import annotations

from model.training.readout_guard import LONG_SCHEDULE_EPOCHS, schedule_warning


def test_warns_for_last_on_a_long_schedule():
    warning = schedule_warning("last", 20)
    assert warning is not None
    assert "weighted" in warning
    assert "8B" in warning and "0.6B" in warning


def test_does_not_warn_for_last_on_a_short_schedule():
    # Every short-schedule arm in research/10 is a `last` arm and must stay
    # reproducible without a warning that suggests changing the recipe.
    assert schedule_warning("last", 5) is None
    assert schedule_warning("last", 1) is None


def test_warns_at_the_documented_threshold():
    assert schedule_warning("last", LONG_SCHEDULE_EPOCHS) is not None
    assert schedule_warning("last", LONG_SCHEDULE_EPOCHS - 1) is None


def test_never_warns_for_the_readouts_that_have_not_collapsed():
    for readout in ("weighted", "mean_last_n"):
        assert schedule_warning(readout, 20) is None


def test_warning_names_both_observed_collapses():
    warning = schedule_warning("last", 10)
    assert "47.51" in warning and "36.43" in warning  # 8B, two seeds
    assert "0.4807" in warning  # 0.6B, after epoch 1
