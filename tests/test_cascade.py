"""Tests for the cascade decision rule (#342)."""

from __future__ import annotations

import pytest

from model.evaluation.cascade import (
    best,
    evaluate,
    prediction,
    probabilities,
    ranked,
    second_pass,
    split_halves,
    threshold_grid,
    top_two,
)


def test_probabilities_sum_to_one_and_survive_large_logits():
    probs = probabilities([1000.0, 999.0, 0.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_top_two_orders_by_probability_and_breaks_ties_by_index():
    first, second, margin = top_two([1.0, 3.0, 2.0])
    assert (first, second) == (1, 2)
    assert margin > 0

    first, second, margin = top_two([1.0, 1.0, 0.0])
    assert (first, second) == (0, 1)
    assert margin == 0.0


def test_ranked_is_stable_on_exact_ties():
    assert ranked([5.0, 5.0, 5.0]) == [0, 1, 2]


def test_second_pass_ties_to_the_smaller_natural_index():
    assert second_pass([2, 1], {1: 0.5, 2: 0.5}) == 1
    assert second_pass([2, 1], {1: 0.4, 2: 0.6}) == 2


def test_prediction_triggers_only_below_the_threshold():
    candidates = [0, 1]
    scores = {0: -0.1, 1: -9.0}
    # Above or at the threshold: no second pass.
    assert prediction(0, candidates, scores, 0.11, 0.1) == (0, False)
    assert prediction(0, candidates, scores, 0.1, 0.1) == (0, False)
    # Below it: the second pass runs and picks index 0 again.
    assert prediction(0, candidates, scores, 0.099, 0.1) == (0, True)
    # The second pass can also flip the answer.
    flipped = {0: -9.0, 1: -0.1}
    assert prediction(0, candidates, flipped, 0.0, 0.1) == (1, True)


def test_evaluate_counts_trigger_fraction_and_objective():
    rows = [
        # triggers, second pass flips to 1 and is right
        {
            "id": "a",
            "type": "choice",
            "gold": 1,
            "base": 0,
            "candidates": [0, 1],
            "scores": {0: -5.0, 1: -1.0},
            "margin": 0.05,
        },
        # does not trigger, base is right
        {
            "id": "b",
            "type": "noul",
            "gold": 0,
            "base": 0,
            "candidates": [0, 1],
            "scores": {0: -1.0, 1: -5.0},
            "margin": 0.5,
        },
    ]
    result = evaluate(rows, 0.1)

    assert result["items"] == 2
    assert result["correct"] == 2
    assert result["base_correct"] == 1
    assert result["triggered"] == 1
    assert result["trigger_fraction"] == pytest.approx(0.5)
    assert result["mean_forwards"] == pytest.approx(2.0)
    assert result["objective"] == pytest.approx(2 / 2 / 2.0)
    assert result["by_type"]["choice"]["triggered"] == 1
    assert result["by_type"]["noul"]["triggered"] == 0


def test_ineligible_items_never_trigger_and_keep_the_base_answer():
    rows = [
        {
            "id": "a",
            "type": "noul",
            "gold": 1,
            "base": 0,
            "candidates": [0, 1],
            "scores": {0: -5.0, 1: -1.0},
            "margin": 0.01,
            "eligible": False,
        },
    ]
    result = evaluate(rows, 0.5)

    assert result["triggered"] == 0
    assert result["correct"] == 0
    assert result["mean_forwards"] == pytest.approx(1.0)


def test_the_eligible_set_overrides_the_row_flag():
    rows = [
        {
            "id": "a",
            "type": "noul",
            "gold": 1,
            "base": 0,
            "candidates": [0, 1],
            "scores": {0: -5.0, 1: -1.0},
            "margin": 0.01,
            "eligible": False,
        },
    ]
    result = evaluate(rows, 0.5, eligible={"a"})

    assert result["triggered"] == 1
    assert result["correct"] == 1


def test_best_takes_the_larger_threshold_on_an_objective_tie():
    curve = [
        {"threshold": 0.1, "objective": 0.7},
        {"threshold": 0.3, "objective": 0.7},
        {"threshold": 0.2, "objective": 0.6},
    ]
    assert best(curve)["threshold"] == 0.3
    assert best(curve)["objective"] == 0.7


def test_threshold_grid_contains_zero_one_and_every_margin():
    rows = [{"margin": 0.2}, {"margin": 0.05}, {"margin": 0.2}]
    grid = threshold_grid(rows)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert grid[1:-1] == [0.05, 0.2]


def test_split_halves_are_disjoint_and_cover_the_ids():
    ids = [f"item-{index}" for index in range(7)]
    for mode in ("even_odd", "first_second"):
        left, right = split_halves(ids, mode)
        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == set(ids)
        assert len(left) + len(right) == len(ids)
    with pytest.raises(ValueError):
        split_halves(ids, "random")
