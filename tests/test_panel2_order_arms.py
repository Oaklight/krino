"""Tests for the panel-2 order arms: content addressing and the seeded mixture."""

from __future__ import annotations

import pytest

from model.evaluation.panel2 import CODES
from model.evaluation.panel2_arms import (
    ARMS,
    HASHED,
    MEAN3,
    POSITIONAL,
    SEEDS,
    content_codes,
    display_codes,
    mixture_prediction,
    natural_codes,
    option_index_for_display,
    probabilities_by_option,
    seeded_codes,
)


def codes_by_option(arm: str, texts, reversed_order: bool, forward: int) -> list[str]:
    """The code of each natural option, read off one displayed prompt."""
    displayed = display_codes(arm, texts, reversed_order)[forward]
    by_option = [""] * len(texts)
    for position, option in enumerate(option_index_for_display(len(texts), reversed_order)):
        by_option[option] = displayed[position]
    return by_option


def test_content_codes_are_a_function_of_the_text_not_the_position():
    texts = ["red", "green", "blue", "amber"]
    shuffled = [texts[2], texts[0], texts[3], texts[1]]
    original = dict(zip(texts, content_codes(texts)))
    moved = dict(zip(shuffled, content_codes(shuffled)))
    assert original == moved
    assert sorted(original.values()) == list(CODES[: len(texts)])


def test_content_codes_are_reversal_invariant():
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    forward = codes_by_option(HASHED, texts, False, 0)
    reversed_order = codes_by_option(HASHED, texts, True, 0)
    assert forward == reversed_order


def test_content_codes_refuse_duplicate_texts():
    with pytest.raises(ValueError):
        content_codes(["same", "other", " same "])
    # A tie has no content-determined order, so the arm must stop rather than
    # fall back to the input order, which would restore the positional effect.
    assert content_codes(["same", "other", "different"]) != []


def test_seeded_codes_are_deterministic_permutations():
    pytest.importorskip("numpy")
    first = seeded_codes(5, 0)
    assert first == seeded_codes(5, 0)
    assert sorted(first) == list(CODES[:5])
    for seed in SEEDS:
        draw = seeded_codes(6, seed)
        assert sorted(draw) == list(CODES[:6])


def test_seeded_codes_keep_each_option_code_across_display_orders():
    pytest.importorskip("numpy")
    texts = ["one", "two", "three", "four"]
    for forward in range(len(SEEDS)):
        forward_codes = codes_by_option(MEAN3, texts, False, forward)
        reversed_codes = codes_by_option(MEAN3, texts, True, forward)
        assert forward_codes == reversed_codes


def test_positional_codes_follow_the_display_order():
    texts = ["one", "two", "three"]
    forward_codes = codes_by_option(POSITIONAL, texts, False, 0)
    reversed_codes = codes_by_option(POSITIONAL, texts, True, 0)
    assert forward_codes == list(CODES[:3])
    assert reversed_codes == list(reversed(CODES[:3]))


def test_mixture_prediction_is_the_argmax_of_the_probability_mean():
    # The first two assignments pick option 0; the third picks option 1.  A
    # majority vote over the three argmaxes returns 0; the probability mean
    # returns 1, and the declared mixture is the probability mean.
    distributions = [
        [0.5, 0.4, 0.1],
        [0.5, 0.4, 0.1],
        [0.1, 0.45, 0.45],
    ]
    assert mixture_prediction(distributions) == 1


def test_probabilities_by_option_maps_display_positions_back():
    # Reversed display: position 0 is natural option 1, position 1 is option 0.
    probabilities = probabilities_by_option([0.0, 3.0], [1, 0])
    assert probabilities[0] > probabilities[1]
    assert probabilities[0] + probabilities[1] == pytest.approx(1.0)


def test_the_three_arms_are_the_declared_three():
    assert ARMS == (POSITIONAL, HASHED, MEAN3)
    assert SEEDS == (0, 1, 2)


def test_natural_codes_returns_one_mapping_per_forward():
    pytest.importorskip("numpy")
    texts = ["a", "b", "c"]
    assert len(natural_codes(POSITIONAL, texts)) == 1
    assert len(natural_codes(HASHED, texts)) == 1
    assert len(natural_codes(MEAN3, texts)) == 3
