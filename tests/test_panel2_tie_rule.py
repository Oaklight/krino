"""Tests for the panel-2 tie rule and the tie count (#347)."""

from __future__ import annotations

from model.evaluation.panel2 import (
    choose_option,
    has_exact_tie,
    natural_argmax,
    option_index_for_display,
    tie_report,
)


def test_choose_option_returns_the_natural_option_index():
    # Display position 0 holds natural option 1, so the answer must be 1.
    # Returning the display argmax unmapped would give 0.
    assert choose_option([5.0, 3.0], [1, 0]) == 1
    assert choose_option([5.0, 3.0], [0, 1]) == 0


def test_choose_option_pins_the_tie_to_the_lowest_natural_index():
    assert natural_argmax([17.0, 17.5, 17.5]) == 1
    # Reversed display: natural 0 and 1 are tied at 17.5 and natural 0 comes
    # first in natural order, so the answer is 0 even though display position
    # 1 is the first maximum on screen.
    assert choose_option([17.0, 17.5, 17.5], [2, 1, 0]) == 0
    assert choose_option([17.0, 17.5, 17.5], [0, 1, 2]) == 1


def test_a_tied_item_keeps_the_same_correctness_under_reversal():
    values = [17.5, 17.5]
    gold = 0
    outcomes = {}
    for reversed_order in (False, True):
        mapping = option_index_for_display(2, reversed_order)
        chosen = choose_option(values, mapping)
        outcomes[reversed_order] = mapping.index(chosen) == mapping.index(gold)
    assert outcomes[False] and outcomes[True]

    # The rule this replaces: first maximum in display order, compared to the
    # gold's display position. On the same item it flips with the order, which
    # is the artefact the natural-index rule removes.
    flipped = {}
    for reversed_order in (False, True):
        mapping = option_index_for_display(2, reversed_order)
        flipped[reversed_order] = natural_argmax(values) == mapping.index(gold)
    assert flipped[False] != flipped[True]


def test_has_exact_tie_uses_the_top_two_values():
    assert has_exact_tie([17.5, 17.5])
    assert has_exact_tie([1.0, 1.0, 0.5])
    assert has_exact_tie([0.5, 1.0, 1.0])
    assert not has_exact_tie([17.5, 17.25])
    assert not has_exact_tie([1.0])


def test_tie_report_counts_per_forward_union_and_correct():
    rows = [
        {"correct": True},
        {"correct": False},
        {"correct": True},
    ]
    flags = [[True, False], [False, True], [False, False]]
    report = tie_report(rows, flags)
    assert report["per_forward"] == [1, 1]
    assert report["union"] == 2
    assert report["correct"] == 1
    assert tie_report([], []) == {"union": 0, "per_forward": [], "correct": 0}
