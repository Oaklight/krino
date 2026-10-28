"""Phase-scoped memory snapshots must pair peaks with peaks.

The logging exists to answer whether the 8B run grew across epochs or merely
fragmented (#161). That answer is only readable if (a) both numbers are peaks
over the same window, and (b) training and eval are not charged to each other.
These tests pin both, torch-free, via injected callables.
"""

from __future__ import annotations

import pytest

from model.training.memory import format_epoch_memory, peak_snapshot

GIB = 1024**3


def _reader(allocated: float, reserved: float, retries: int = 0):
    """Fake torch.cuda accessors in GiB, plus a reset counter."""
    state = {"resets": 0}

    def reset():
        state["resets"] += 1

    return state, (
        lambda: allocated * GIB,
        lambda: reserved * GIB,
        lambda: {"num_alloc_retries": retries},
        reset,
    )


def test_snapshot_pairs_peaks_and_resets_after_reading():
    state, fns = _reader(allocated=30.0, reserved=40.0, retries=2)
    snap = peak_snapshot(*fns)

    assert snap["peak_allocated"] == pytest.approx(30.0)
    assert snap["peak_reserved"] == pytest.approx(40.0)
    assert snap["fragmentation"] == pytest.approx(10.0)
    assert snap["alloc_retries_total"] == 2
    assert state["resets"] == 1, "the reset must happen so the next phase starts empty"


def test_reserved_below_allocated_cannot_produce_negative_fragmentation():
    """Mixing a peak with a point sample used to make reserved < allocated."""
    _, fns = _reader(allocated=40.0, reserved=39.9)
    snap = peak_snapshot(*fns)
    assert snap["fragmentation"] == 0.0


def test_each_call_resets_so_phases_do_not_accumulate():
    state, fns = _reader(allocated=10.0, reserved=12.0)
    peak_snapshot(*fns)
    peak_snapshot(*fns)
    assert state["resets"] == 2


def test_format_reports_both_phases():
    train = {"peak_allocated": 30.0, "peak_reserved": 40.0, "alloc_retries_total": 0}
    eval_ = {"peak_allocated": 34.0, "peak_reserved": 41.0, "alloc_retries_total": 0}
    text = format_epoch_memory(train, eval_)
    assert "train=30.0/40.0G" in text
    assert "eval=34.0/41.0G" in text
    assert "alloc_retries" not in text


def test_format_flags_allocator_retries():
    """A retry means the allocator had to defragment — the pre-OOM signal."""
    train = {"peak_allocated": 30.0, "peak_reserved": 40.0, "alloc_retries_total": 3}
    assert "alloc_retries=3" in format_epoch_memory(train, None)


def test_format_is_empty_when_memory_is_not_tracked():
    assert format_epoch_memory(None, None) == ""
