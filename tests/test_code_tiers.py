"""Tests for the code-tier design (#350 follow-up)."""

from __future__ import annotations

import random

import pytest

from model.evaluation.code_tiers import (
    ALPHABET,
    K,
    TIER_SLOTS,
    assign_codes,
    bootstrap_ci,
    evaluate_pairs,
    letter_positions,
    mcnemar_p,
    paired_codes,
    reading,
    subsample,
    swap_position,
    tier_positions,
)


def tier_codes() -> list[str]:
    return ["Ä", "È", "Ü", "İ", "ž", "Ấ"]


def test_slot_positions_always_contain_the_gold():
    for gold in (0, 1, 52, 53, 57):
        slots = tier_positions(gold)
        assert len(slots) == TIER_SLOTS
        assert gold in slots
        assert len(letter_positions(gold)) == len(ALPHABET)


def test_swap_position_is_the_smallest_letter_position_not_position_zero():
    # With the gold at position 0, a literal "swap with position 0" would be a
    # no-op and the two conditions would be identical.
    assert swap_position(0) == 1
    assert swap_position(5) == 0
    assert swap_position(57) == 0


def test_the_two_conditions_differ_only_at_the_gold_and_the_swap():
    gold = 17
    in_tier = assign_codes(gold, tier_codes(), True)
    letter = assign_codes(gold, tier_codes(), False)
    assert sorted(in_tier) == sorted(letter)
    changed = [i for i in range(K) if in_tier[i] != letter[i]]
    assert changed == [swap_position(gold), gold]
    assert in_tier[gold] in tier_codes()
    assert letter[gold] in ALPHABET


def test_a_pair_with_a_gold_at_position_zero_still_has_distinct_gold_codes():
    pair = paired_codes(0, tier_codes())
    assert pair[True][0] != pair[False][0]
    assert pair[True][0] in tier_codes()
    assert pair[False][0] in ALPHABET


def test_every_gold_position_gives_a_distinct_gold_code():
    for gold in range(K):
        pair = paired_codes(gold, tier_codes())
        assert pair, f"gold at {gold} produced an empty pair"
        assert pair[True][gold] != pair[False][gold]


def test_duplicate_tier_codes_are_rejected():
    with pytest.raises(ValueError):
        assign_codes(3, ["Ä", "Ä", "Ü", "İ", "ž", "Ấ"], True)


def test_subsample_keeps_the_gold_and_is_deterministic():
    criteria = {f"key{i}": f"text {i}" for i in range(80)}
    texts_a, gold_a = subsample(criteria, "key7", K, random.Random("342:a"))
    texts_b, gold_b = subsample(criteria, "key7", K, random.Random("342:a"))
    assert texts_a == texts_b and gold_a == gold_b
    assert len(texts_a) == K
    assert texts_a[gold_a] == "text 7"
    assert len(set(texts_a)) == K


def test_reading_is_three_way_at_the_declared_bounds():
    assert reading(-10.0, 0.01, -20.0) == "rejected"
    assert reading(-10.0, 0.20, -20.0) == "undecided"
    assert reading(-4.9, 0.50, -9.0) == "adoptable"
    assert reading(-4.9, 0.50, -11.0) == "undecided"
    assert reading(-5.0, 0.50, -9.0) == "undecided"


def test_evaluate_pairs_counts_discordance_and_direction():
    pairs = [(True, True)] * 10 + [(False, True)] * 8 + [(True, False)] * 2 + [(False, False)] * 4
    cell = evaluate_pairs(pairs)
    assert cell["items"] == 24
    assert cell["gold_in_tier_correct"] == 12
    assert cell["letter_gold_correct"] == 18
    assert cell["discordant"] == {
        "tier_correct_letter_wrong": 2,
        "tier_wrong_letter_correct": 8,
    }
    assert cell["difference_pp"] == pytest.approx(100 * (2 - 8) / 24)
    assert cell["reading"] == "undecided"


def test_mcnemar_matches_the_exact_symbolic_values():
    assert mcnemar_p(6, 0) == pytest.approx(0.03125)
    assert mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 3) == 1.0


def test_bootstrap_is_deterministic_and_brackets_zero():
    pairs = [(True, True)] * 60 + [(False, True)] * 15 + [(True, False)] * 15 + [(False, False)] * 10
    first = bootstrap_ci(pairs)
    second = bootstrap_ci(pairs)
    assert first == second
    assert first["low"] < 0 < first["high"]
