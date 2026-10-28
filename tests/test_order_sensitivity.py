"""Tests for the pure option-order sensitivity helpers.

These run without torch, matching the rest of the metric tests in CI.
"""

from __future__ import annotations

import pytest

from model.evaluation.order_sensitivity import (
    make_permutation,
    option_count_bucket,
    remap_probe_probs,
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
