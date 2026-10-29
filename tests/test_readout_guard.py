"""Tests for the long-schedule/readout guard."""

from __future__ import annotations

import json
from pathlib import Path

from model.training.readout_guard import (
    FRAGILE_PARAMETER_BILLIONS,
    LONG_SCHEDULE_EPOCHS,
    backbone_billions,
    schedule_warning,
)

HISTORY = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "experiments"
    / "arch_06b_loraAll_last_10ep_s123_epochs.json"
)


def test_warns_for_last_on_a_long_schedule():
    warning = schedule_warning("last", 20, "Qwen/Qwen3-Reranker-4B")
    assert warning is not None
    assert "weighted" in warning


def test_warns_for_last_on_a_large_backbone_at_the_ordinary_length():
    # The 8B collapse happened at five epochs, so it is a scale effect and the
    # schedule rule alone would miss it.
    warning = schedule_warning("last", 5, "Qwen/Qwen3-Reranker-8B")
    assert warning is not None
    assert "8B" in warning


def test_is_silent_for_a_small_backbone_on_a_short_schedule():
    # 5-epoch arms are left unwarned by choice: they are the arms research/10
    # published, and nothing here changes a run. The 8B case above is the one
    # documented gap in the schedule rule.
    assert schedule_warning("last", 5, "Qwen/Qwen3-Reranker-0.6B") is None
    assert schedule_warning("last", 5, "Qwen/Qwen3-Reranker-4B") is None
    assert schedule_warning("last", 1) is None


def test_boundaries_are_inclusive_where_the_failures_were_observed():
    assert schedule_warning("last", LONG_SCHEDULE_EPOCHS) is not None
    assert schedule_warning("last", LONG_SCHEDULE_EPOCHS - 1) is None
    assert FRAGILE_PARAMETER_BILLIONS == 8.0


def test_never_warns_for_the_readouts_that_have_not_collapsed():
    for readout in ("weighted", "mean_last_n"):
        assert schedule_warning(readout, 20, "Qwen/Qwen3-Reranker-8B") is None


def test_backbone_size_is_read_from_the_name_and_unknown_names_are_ignored():
    assert backbone_billions("Qwen/Qwen3-Reranker-8B") == 8.0
    assert backbone_billions("Qwen/Qwen3-Reranker-0.6B") == 0.6
    assert backbone_billions("some/model-with-no-size") is None
    assert backbone_billions(None) is None
    # An unrecognised name must not trip the scale rule.
    assert schedule_warning("last", 5, "some/model-with-no-size") is None


def test_one_unit_throughout_the_message():
    # Percentages only: an earlier draft mixed 47.51 with 0.4807 and read as two
    # different scales.
    warning = schedule_warning("last", 20, "Qwen/Qwen3-Reranker-4B")
    assert "0.4807" not in warning
    assert "48.07%" in warning


def test_the_message_cites_the_committed_history():
    # Both failures are named, and the 0.6B numbers match the committed artifact
    # so the threshold cannot drift away from its evidence.
    warning = schedule_warning("last", 20)
    assert "47.51%" in warning and "36.43%" in warning
    assert "48.07%" in warning and "31.0-36.5%" in warning

    history = json.loads(HISTORY.read_text())
    epochs = history["epochs"]
    assert epochs[0]["eval_accuracy"] == 0.4807
    later = [e["eval_accuracy"] for e in epochs[1:]]
    assert min(later) == 0.3097 and max(later) == 0.3654
    assert history["readout"] == "last"
