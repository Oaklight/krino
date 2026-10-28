"""Tests for the pure option-order sensitivity helpers.

These run without torch, matching the rest of the metric tests in CI.
"""

from __future__ import annotations

import pytest

from model.evaluation.order_sensitivity import (
    chance_accuracy,
    duplicate_option_texts,
    majority_class_accuracy,
    make_permutation,
    nested_option_subsets,
    normalized_accuracy,
    option_count,
    option_count_bucket,
    remap_probe_probs,
    select_items_by_option_limit,
    tie_indices,
    tie_stable_argmax,
    total_variation,
)


class TestOptionCountBucket:
    @pytest.mark.parametrize(
        ("n_options", "bucket"),
        [
            (1, "2"),
            (2, "2"),
            (3, "3-5"),
            (5, "3-5"),
            (6, "6-20"),
            (20, "6-20"),
            (21, ">20"),
            (255, ">20"),
        ],
    )
    def test_boundaries(self, n_options, bucket):
        assert option_count_bucket(n_options) == bucket


class TestTotalVariation:
    def test_identical_distributions_are_zero(self):
        assert total_variation(
            {"a": 0.7, "b": 0.3}, {"a": 0.7, "b": 0.3}
        ) == pytest.approx(0.0)

    def test_disjoint_distributions_are_one(self):
        assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)

    def test_half_the_mass_moving_is_half(self):
        assert total_variation(
            {"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}
        ) == pytest.approx(0.5)

    def test_missing_labels_count_as_zero(self):
        assert total_variation({"a": 0.5}, {"a": 0.5, "b": 0.0}) == pytest.approx(0.0)


class TestRemapProbeProbs:
    def test_reorders_probabilities_onto_original_labels(self):
        # Probe order was [c, a, b], so position 0 belongs to c, 1 to a, 2 to b.
        remapped = remap_probe_probs([0.6, 0.3, 0.1], ["c", "a", "b"])
        assert remapped == {"c": 0.6, "a": 0.3, "b": 0.1}

    def test_values_are_floats(self):
        remapped = remap_probe_probs([1], [0])
        assert isinstance(remapped[0], float)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            remap_probe_probs([0.5, 0.5], ["a"])


class TestMakePermutation:
    def test_is_a_valid_permutation(self):
        permutation = make_permutation(7, seed=42, item_id="item-1", tag="order")
        assert sorted(permutation) == list(range(7))

    def test_is_deterministic(self):
        first = make_permutation(7, seed=42, item_id="item-1", tag="order")
        second = make_permutation(7, seed=42, item_id="item-1", tag="order")
        assert first == second

    def test_tags_produce_different_orders(self):
        order = make_permutation(20, seed=42, item_id="item-1", tag="order")
        text = make_permutation(20, seed=42, item_id="item-1", tag="text")
        assert order != text


class _Item:
    """Minimal stand-in for TypedQuestion in the torch-free tests."""

    def __init__(self, question, teacher_probs=None):
        self.question = question
        self.teacher_probs = teacher_probs


def _choice_item(n_options):
    return _Item(
        {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {f"k{i}": f"text {i}" for i in range(n_options)},
        }
    )


class TestOptionCount:
    def test_choice_counts_criteria(self):
        assert (
            option_count({"type": "choice", "criteria": {"a": "", "b": "", "c": ""}})
            == 3
        )

    def test_score_counts_levels(self):
        assert option_count({"type": "score", "criteria": ["a", "b", "c", "d"]}) == 4

    def test_binary_noul_is_two(self):
        assert option_count({"type": "noul"}) == 2

    def test_augmented_noul_uses_option_list_with_teacher_probs(self):
        question = {
            "type": "noul",
            "augmented_options": {f"k{i}": f"text {i}" for i in range(5)},
        }
        assert option_count(question, {f"k{i}": 0.2 for i in range(5)}) == 5

    def test_augmented_noul_stays_binary_when_teacher_probs_are_flat(self):
        question = {"type": "noul", "augmented_options": {"true": "", "false": ""}}
        assert option_count(question, {"jev": {"true": 0.9, "false": 0.1}}) == 2


class TestOptionLimit:
    def test_skips_items_above_the_limit_and_counts_them(self):
        kept, skipped = select_items_by_option_limit(
            [_choice_item(2), _choice_item(300)], max_options=255
        )
        assert len(kept) == 1
        assert skipped == 1
        assert option_count(kept[0].question) == 2

    def test_zero_disables_the_guard(self):
        kept, skipped = select_items_by_option_limit(
            [_choice_item(2), _choice_item(300)], max_options=0
        )
        assert len(kept) == 2
        assert skipped == 0

    def test_item_exactly_at_the_limit_is_kept(self):
        kept, skipped = select_items_by_option_limit(
            [_choice_item(255)], max_options=255
        )
        assert len(kept) == 1
        assert skipped == 0


class TestTieStableArgmax:
    def test_breaks_exact_ties_by_order_key(self):
        assert tie_stable_argmax([0.4, 0.4, 0.2], [2, 0, 1]) == 1

    def test_unique_maximum_ignores_order_keys(self):
        assert tie_stable_argmax([0.1, 0.7, 0.2], [0, 1, 2]) == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            tie_stable_argmax([0.5, 0.5], [0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tie_stable_argmax([], [])


class TestTieIndices:
    def test_detects_exact_tie(self):
        assert tie_indices([0.4, 0.4, 0.2]) == [0, 1]

    def test_no_tie(self):
        assert tie_indices([0.1, 0.9]) == [1]


class TestDuplicateOptionTexts:
    def test_reports_repeated_texts(self):
        assert duplicate_option_texts(["a", "b", "a", "c", "b", "b"]) == {
            "a": 2,
            "b": 3,
        }

    def test_empty_when_unique(self):
        assert duplicate_option_texts(["a", "b", "c"]) == {}


class TestChanceAndNormalized:
    def test_chance_is_one_over_options(self):
        assert chance_accuracy(2) == pytest.approx(0.5)
        assert chance_accuracy(77) == pytest.approx(1 / 77)
        assert chance_accuracy(0) == 0.0

    def test_normalized_accuracy_endpoints(self):
        assert normalized_accuracy(0.5, 0.5) == pytest.approx(0.0)
        assert normalized_accuracy(1.0, 0.5) == pytest.approx(1.0)

    def test_normalized_accuracy_many_options(self):
        # 0.344 against a 1/77 chance is 33.5% of the way to perfect.
        assert normalized_accuracy(0.344, 1 / 77) == pytest.approx(
            (0.344 - 1 / 77) / (1 - 1 / 77)
        )


class TestNestedOptionSubsets:
    def test_every_subset_contains_the_gold_option(self):
        subsets = nested_option_subsets(20, 7, [2, 5, 20], seed=42, item_id="x")
        for size, subset in subsets.items():
            assert len(subset) == size
            assert len(set(subset)) == size
            assert 7 in subset
            assert all(0 <= index < 20 for index in subset)

    def test_is_deterministic(self):
        first = nested_option_subsets(20, 7, [2, 5], seed=42, item_id="x")
        second = nested_option_subsets(20, 7, [2, 5], seed=42, item_id="x")
        assert first == second

    def test_out_of_range_sizes_are_skipped(self):
        subsets = nested_option_subsets(4, 1, [1, 2, 5], seed=42, item_id="x")
        assert sorted(subsets) == [2]

    def test_full_size_is_all_options(self):
        subsets = nested_option_subsets(5, 2, [5], seed=42, item_id="x")
        assert sorted(subsets[5]) == [0, 1, 2, 3, 4]


class TestMajorityClassAccuracy:
    def test_all_same_label_is_one(self):
        assert majority_class_accuracy(["no", "no", "no"]) == 1.0

    def test_matches_the_jevbench_noul_case(self):
        # 39 no / 35 yes: a constant "no" predictor scores exactly 52.70%.
        labels = ["no"] * 39 + ["yes"] * 35
        assert majority_class_accuracy(labels) == pytest.approx(39 / 74)

    def test_balanced_labels_are_one_half(self):
        assert majority_class_accuracy(["a", "b", "a", "b"]) == 0.5

    def test_empty_is_zero(self):
        assert majority_class_accuracy([]) == 0.0
