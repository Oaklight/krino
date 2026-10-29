"""Tests for the scorer blend, its buckets, and the sampling rule.

Pure Python, so it runs without torch. What the tests pin is what a wrong
implementation would quietly change: the endpoints of the sweep must be the two
scorers themselves, the scale must come from the fitting set alone, ties must not
depend on option order, and the per-source sampling must be deterministic and
keep a source's short items even when it is mostly long ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from model.evaluation.scorer_blend import (
    LONG_MIN,
    SHORT_MAX,
    blended_argmax,
    bucket_of,
    margin,
    sample_choice_items,
    spread,
)


@dataclass
class _Item:
    question: dict


def _items(count: int, n_options: int) -> list[_Item]:
    return [
        _Item({"criteria": {str(index): f"text {position}" for index in range(n_options)}})
        for position in range(count)
    ]


class TestMargin:
    def test_is_top_minus_runner_up(self):
        assert margin([1.0, 5.0, 3.0]) == 2.0

    def test_single_option_has_no_margin(self):
        assert margin([4.0]) == 0.0

    def test_tied_top_scores_zero(self):
        assert margin([2.0, 2.0, 1.0]) == 0.0


class TestSpread:
    def test_uses_the_median(self):
        assert spread([1.0, 2.0, 3.0]) == 2.0

    def test_ignores_non_positive_margins(self):
        # Ties carry no information about scale; letting them in drags the
        # median toward zero and then tiny differences dominate the blend.
        assert spread([0.0, 0.0, 4.0]) == 4.0

    def test_all_ties_fall_back_to_one(self):
        assert spread([0.0, 0.0]) == 1.0

    def test_empty_falls_back_to_one(self):
        assert spread([]) == 1.0


class TestBlendedArgmax:
    def test_lambda_zero_is_the_head_argmax(self):
        assert blended_argmax([0.0, 5.0], [9.0, 0.0], 1.0, 1.0, 0.0) == 1

    def test_large_lambda_is_the_native_argmax(self):
        assert blended_argmax([0.0, 5.0], [9.0, 0.0], 1.0, 1.0, 1000.0) == 0

    def test_the_scales_normalise_the_two_sources(self):
        # The head's margin is 5 and the native's is 0.5; at lam=1 with each
        # scorer divided by its own spread, the head still wins. Without the
        # rescaling the raw native differences would swamp it.
        head, native = [0.0, 5.0], [0.0, 0.5]
        assert blended_argmax(head, native, 5.0, 0.5, 1.0) == 1

    def test_ties_break_toward_the_lower_index(self):
        assert blended_argmax([1.0, 1.0], [2.0, 2.0], 1.0, 1.0, 1.0) == 0

    def test_mismatched_lengths_are_an_error(self):
        with pytest.raises(ValueError):
            blended_argmax([1.0], [1.0, 2.0], 1.0, 1.0, 1.0)

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            blended_argmax([], [], 1.0, 1.0, 1.0)


class TestBucketOf:
    def test_short_boundary(self):
        assert bucket_of(SHORT_MAX) == f"<={SHORT_MAX}"
        assert bucket_of(SHORT_MAX + 1) != f"<={SHORT_MAX}"

    def test_long_boundary(self):
        assert bucket_of(LONG_MIN) == f">={LONG_MIN}"
        assert bucket_of(LONG_MIN - 1) != f">={LONG_MIN}"

    def test_every_option_count_lands_in_exactly_one_bucket(self):
        buckets = {bucket_of(count) for count in range(1, 300)}
        assert len({bucket_of(count) for count in range(1, SHORT_MAX + 1)}) == 1


class TestSampleChoiceItems:
    def test_caps_short_items(self):
        sampled = sample_choice_items(_items(100, 3), per_source=10, long_items=0, rng=random.Random(0))
        assert len(sampled) == 10

    def test_caps_long_items_with_their_own_limit(self):
        sampled = sample_choice_items(_items(100, 80), per_source=10, long_items=4, rng=random.Random(0))
        assert len(sampled) == 4

    def test_a_mostly_long_source_still_contributes_its_short_items(self):
        items = _items(100, 80) + _items(5, 2)
        sampled = sample_choice_items(items, per_source=10, long_items=3, rng=random.Random(0))
        short = [item for item in sampled if len(item.question["criteria"]) < LONG_MIN]
        assert len(short) == 5
        assert len(sampled) == 8

    def test_is_deterministic_for_a_seed(self):
        items = _items(50, 4)
        first = sample_choice_items(items, 10, 0, random.Random(7))
        again = sample_choice_items(items, 10, 0, random.Random(7))
        assert [id(item) for item in first] == [id(item) for item in again]

    def test_varies_across_seeds(self):
        items = _items(50, 4)
        first = sample_choice_items(items, 10, 0, random.Random(7))
        other = sample_choice_items(items, 10, 0, random.Random(8))
        assert [id(item) for item in first] != [id(item) for item in other]

    def test_a_source_under_the_cap_is_returned_whole(self):
        items = _items(4, 3)
        sampled = sample_choice_items(items, 60, 150, random.Random(0))
        assert len(sampled) == len(items)

    def test_missing_criteria_falls_back_to_short(self):
        sampled = sample_choice_items(
            [_Item({"type": "choice"})], per_source=5, long_items=0, rng=random.Random(0)
        )
        assert len(sampled) == 1
