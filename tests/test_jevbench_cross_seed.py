"""Tests for the cross-checkpoint JevBench comparison helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "model" / "scripts" / "jevbench_cross_seed.py"
spec = importlib.util.spec_from_file_location("jevbench_cross_seed", SCRIPT)
cross_seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_seed)


def test_argmax_takes_the_highest_and_keeps_the_first_tie():
    assert cross_seed.argmax([0.1, 0.9, 0.2]) == 1
    assert cross_seed.argmax([0.5, 0.5, 0.1]) == 0
    assert cross_seed.argmax([-3.0, -1.0]) == 1


def test_summarise_counts_both_either_and_disagreements():
    first = [True, True, False, False]
    second = [True, False, True, False]
    table = cross_seed.summarise(first, second)
    assert table["n"] == 4
    assert table["correct"] == [2, 2]
    assert table["accuracy"] == [0.5, 0.5]
    assert table["both_right"] == 1
    assert table["either_right"] == 3
    assert table["either_accuracy"] == 0.75
    assert table["disagree"] == 2


def test_independent_overlap_is_the_perfect_agreement_null():
    # Two answerers at 50% on four items would agree on one of them if they
    # answered independently; it is the number the observed overlap is read
    # against, not a floor either has to clear.
    table = cross_seed.summarise([True, True, False, False], [True, True, False, False])
    assert table["independent_overlap"] == 1.0
    assert table["both_right"] == 2


def test_summarise_handles_a_perfect_and_an_empty_pair():
    assert cross_seed.summarise([True, True], [True, True])["either_right"] == 2
    assert cross_seed.summarise([False, False], [False, False])["both_right"] == 0


def test_outcomes_follow_each_readout_independently():
    records = [
        {"head": [2.0, 0.0], "native": [0.0, 2.0], "gold": 1},
        {"head": [2.0, 0.0], "native": [2.0, 0.0], "gold": 1},
    ]
    assert cross_seed.outcomes(records, "head") == [False, False]
    assert cross_seed.outcomes(records, "native") == [True, False]


def test_align_rejects_a_record_count_that_does_not_match_the_items():
    items = [{"id": "hard-a", "question": {"type": "choice", "criteria": ["x", "y"]}, "labels": ["x", "y"]}]
    with pytest.raises(SystemExit, match="positional mapping"):
        cross_seed.align([], items, "checkpoint 0")


def test_align_rejects_records_that_are_out_of_order():
    items = [
        {"id": "hard-a", "question": {"type": "choice", "criteria": ["x", "y"]}, "labels": ["x", "y"]},
        {"id": "hard-b", "question": {"type": "choice", "criteria": ["x", "y"]}, "labels": ["x", "y"]},
    ]
    records = [{"k": 2}, {"k": 3}]
    with pytest.raises(SystemExit, match="not in the same order"):
        cross_seed.align(records, items, "checkpoint 1")


def test_align_accepts_matching_records():
    items = [{"id": "hard-a", "question": {"type": "choice", "criteria": ["x", "y"]}, "labels": ["x", "y"]}]
    cross_seed.align([{"k": 2}], items, "checkpoint 0")
