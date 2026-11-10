"""Tests for the blocking readout's design (#356).

The blocking frontier's every number rests on four pieces of logic: how the
option list is cut into blocks, how the per-item shuffle is seeded, whether a
scheme's correctness includes the gold winning its block, and the two-factor
decomposition. These tests pin each one, and each is written so that breaking
the behaviour turns it red.
"""

from __future__ import annotations

import pytest

from model.evaluation.blocking import (
    BLOCK_SIZES,
    K,
    blocks_of,
    decompose,
    gold_block,
    position_bin,
    scheme_a_correct,
    scheme_b_correct,
    shuffled_blocks,
    shuffled_order,
)


def test_blocks_of_covers_every_index_exactly_once():
    for size in BLOCK_SIZES:
        blocks = blocks_of(K, size)
        flat = [index for block in blocks for index in block]
        assert flat == list(range(K))
        assert len(set(flat)) == K
        assert len(blocks) == -(-K // size)
        assert all(len(block) <= size for block in blocks)
        assert all(block for block in blocks)


def test_the_last_block_is_short_when_k_is_not_a_multiple():
    # 150 / 52 -> three blocks, the last of 46; 150 / 32 -> five, the last of 22.
    assert [len(block) for block in blocks_of(150, 52)] == [52, 52, 46]
    assert [len(block) for block in blocks_of(150, 32)] == [32, 32, 32, 32, 22]
    assert [len(block) for block in blocks_of(150, 26)] == [26, 26, 26, 26, 26, 20]
    # An exact multiple has no short block.
    assert [len(block) for block in blocks_of(150, 25)] == [25] * 6


def test_blocks_of_rejects_a_non_positive_size():
    with pytest.raises(ValueError):
        blocks_of(150, 0)
    with pytest.raises(ValueError):
        blocks_of(0, 52)


def test_every_gold_is_in_exactly_one_block():
    for size in BLOCK_SIZES:
        blocks = blocks_of(K, size)
        for gold in range(K):
            holders = [index for index, block in enumerate(blocks) if gold in block]
            assert len(holders) == 1
            assert gold_block(blocks, gold) == holders[0]
    with pytest.raises(ValueError):
        gold_block([[0, 1]], 5)


def test_the_shuffle_is_deterministic_for_one_item_id():
    for size in BLOCK_SIZES:
        first = shuffled_blocks("clinc150-test-00042", size)
        second = shuffled_blocks("clinc150-test-00042", size)
        assert first == second
        # The blocks are consecutive slices of one per-item permutation.
        flat = [index for block in first for index in block]
        assert flat == shuffled_order("clinc150-test-00042")
        # The same permutation at another block size is cut differently, not
        # resampled.
        assert flat == [
            index for block in shuffled_blocks("clinc150-test-00042", 7) for index in block
        ]


def test_the_shuffle_depends_on_the_item_id_and_is_not_the_identity():
    assert shuffled_order("item-a") != shuffled_order("item-b")
    assert sorted(shuffled_order("item-a")) == list(range(K))
    identity = list(range(K))
    differing = sum(
        shuffled_order(f"clinc150-test-{index:05d}") != identity
        for index in range(50)
    )
    # An accidental no-op shuffle would make this zero, and the "not class-order
    # luck" conclusion would be empty.
    assert differing == 50


def test_scheme_b_is_wrong_when_the_gold_lost_its_block():
    # The final pass sees only the block winners. Even a final pick that returns
    # the gold's index cannot recover an option that never reached the list.
    assert not scheme_b_correct(gold=3, winners=[2, 5], final_pick=3)
    assert scheme_b_correct(gold=2, winners=[2, 5], final_pick=2)
    assert not scheme_b_correct(gold=2, winners=[2, 5], final_pick=5)


def test_scheme_a_compares_the_winner_logits_and_ties_to_the_earlier_block():
    assert scheme_a_correct(2, [2, 5], [1.0, 1.0])
    assert not scheme_a_correct(5, [2, 5], [1.0, 1.0])
    assert scheme_a_correct(5, [2, 5], [1.0, 2.0])
    with pytest.raises(ValueError):
        scheme_a_correct(0, [0], [1.0, 2.0])


def test_decompose_reproduces_the_end_to_end_count():
    rows = [
        {"won_block": True, "correct": True},
        {"won_block": True, "correct": False},
        {"won_block": False, "correct": False},
        {"won_block": True, "correct": True},
    ]
    result = decompose(rows)
    assert result["items"] == 4
    assert result["won_block"] == 3
    assert result["won_block_rate"] == pytest.approx(0.75)
    assert result["final_given_won"] == pytest.approx(2 / 3)
    assert result["product"] == pytest.approx(2 / 4)
    assert result["product"] == pytest.approx(
        result["won_block_rate"] * result["final_given_won"]
    )


def test_decompose_is_defined_on_an_empty_cell():
    result = decompose([])
    assert result["items"] == 0
    assert result["product"] == 0.0
    assert result["final_given_won"] == 0.0


def test_position_bin_boundaries():
    # 52 options: first is local < 10.4, last is local >= 41.6.
    assert position_bin(0, 52) == "first"
    assert position_bin(10, 52) == "first"
    assert position_bin(11, 52) == "middle"
    assert position_bin(41, 52) == "middle"
    assert position_bin(42, 52) == "last"
    assert position_bin(51, 52) == "last"
    # 26 options: first is local < 5.2, last is local >= 20.8.
    assert position_bin(5, 26) == "first"
    assert position_bin(6, 26) == "middle"
    assert position_bin(20, 26) == "middle"
    assert position_bin(21, 26) == "last"
    with pytest.raises(ValueError):
        position_bin(52, 52)
    with pytest.raises(ValueError):
        position_bin(-1, 52)
