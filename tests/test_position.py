"""Tests for the position probes' design (#357).

Three pieces decide every number the probes report: how a position is bucketed,
how the per-item permutation is seeded, and whether the K=52 reproduction gate
stops a drifted run. Each test is written so that breaking the behaviour turns
it red.
"""

from __future__ import annotations

import pytest

from model.evaluation.position import GATES, ITEMS, bucket, check_gate, permuted


def test_every_position_lands_in_exactly_one_bucket():
    for k in (7, 14, 28, 52, 80):
        assigned = [bucket(position, k) for position in range(k)]
        assert len(assigned) == k
        assert set(assigned) <= set(range(5))
        assert assigned == sorted(assigned)
        assert assigned[0] == 0
        assert assigned[-1] == 4
        # Every bucket is non-empty once there are at least five positions.
        assert {b for b in range(5) if b in assigned} == set(range(5))


def test_bucket_boundaries_when_k_is_not_a_multiple_of_five():
    # 52 / 5: 0-10, 11-20, 21-31, 32-41, 42-51.
    assert [bucket(p, 52) for p in (0, 10, 11, 20, 21, 31, 32, 41, 42, 51)] == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4
    ]
    # 7 / 5: two positions in bucket 0 and two in bucket 2, the last clipped.
    assert [bucket(p, 7) for p in range(7)] == [0, 0, 1, 2, 2, 3, 4]
    # 14 / 5: 0-2, 3-5, 6-8, 9-11, 12-13.
    assert [bucket(p, 14) for p in (0, 2, 3, 5, 6, 8, 9, 11, 12, 13)] == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4
    ]
    with pytest.raises(ValueError):
        bucket(52, 52)
    with pytest.raises(ValueError):
        bucket(-1, 52)
    with pytest.raises(ValueError):
        bucket(0, 0)


def test_the_permutation_is_deterministic_per_item_and_index():
    texts = [f"option {index}" for index in range(12)]
    first, first_gold = permuted(texts, 3, "item-a", 0)
    second, second_gold = permuted(texts, 3, "item-a", 0)
    assert (first, first_gold) == (second, second_gold)
    assert sorted(first) == sorted(texts)
    assert first[first_gold] == texts[3]


def test_the_permutation_changes_with_the_item_and_the_index():
    texts = [f"option {index}" for index in range(20)]
    assert permuted(texts, 0, "item-a", 0)[0] != permuted(texts, 0, "item-b", 0)[0]
    assert permuted(texts, 0, "item-a", 0)[0] != permuted(texts, 0, "item-a", 1)[0]


def test_the_permutation_is_not_the_identity_on_most_items():
    texts = [f"option {index}" for index in range(52)]
    identity = list(texts)
    differing = sum(
        permuted(texts, 0, f"banking77-test-{index:05d}", 0)[0] != identity
        for index in range(50)
    )
    # An accidental no-op shuffle would make this zero, and the natural-rank
    # reading would silently become a display-rank reading.
    assert differing == 50


def gate_rows(dataset: str, correct: int, items: int = ITEMS) -> list[dict]:
    return [
        {"cell": "natural_k52", "correct": index < correct}
        for index in range(items)
    ]


def test_the_gate_passes_on_the_recorded_counts():
    assert check_gate(gate_rows("banking77", GATES["banking77"]), "banking77") == {
        "items": ITEMS,
        "correct": GATES["banking77"],
    }
    assert check_gate(gate_rows("clinc150", GATES["clinc150"]), "clinc150")["correct"] == 146


def test_the_gate_fails_loudly_when_the_cell_does_not_reproduce():
    with pytest.raises(SystemExit):
        check_gate(gate_rows("banking77", GATES["banking77"] - 1), "banking77")
    with pytest.raises(SystemExit):
        check_gate(gate_rows("clinc150", GATES["clinc150"] + 1), "clinc150")
    with pytest.raises(SystemExit):
        check_gate(gate_rows("banking77", GATES["banking77"], items=ITEMS - 1), "banking77")


def test_the_gate_ignores_other_cells():
    rows = gate_rows("banking77", GATES["banking77"])
    rows += [{"cell": "perm_k52_1", "correct": False} for _ in range(10)]
    assert check_gate(rows, "banking77")["items"] == ITEMS
