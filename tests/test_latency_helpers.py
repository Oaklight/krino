"""Tests for the latency summary helpers.

Torch-free, so they run on a bare interpreter and in CI.
"""

from __future__ import annotations

import pytest

from model.evaluation.latency import (
    DEFAULT_OPTION_COUNTS,
    make_options,
    percentile,
    summarise,
)


class TestMakeOptions:
    def test_requested_count(self):
        for n in (1, 2, 5, 20, 256):
            assert len(make_options(n)) == n

    def test_options_are_distinct(self):
        options = make_options(256)
        assert len(set(options)) == 256

    def test_first_five_are_the_bare_stems(self):
        assert make_options(5) == [
            "Ship now and accept the known failure",
            "Delay the release by one week",
            "Ship behind a feature flag",
            "Roll back to the previous build",
            "Extend the freeze and fix the regression",
        ]

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            make_options(0)

    def test_deterministic(self):
        assert make_options(32) == make_options(32)


class TestPercentile:
    def test_nearest_rank(self):
        values = [float(i) for i in range(1, 11)]  # 1..10
        assert percentile(values, 0.5) == 5.0
        assert percentile(values, 0.95) == 10.0
        assert percentile(values, 1.0) == 10.0

    def test_unsorted_input(self):
        assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0

    def test_single_sample(self):
        assert percentile([0.25], 0.95) == 0.25

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            percentile([], 0.5)

    def test_fraction_bounds(self):
        with pytest.raises(ValueError):
            percentile([1.0], 0.0)
        with pytest.raises(ValueError):
            percentile([1.0], 1.5)


class TestSummarise:
    def test_seconds_become_milliseconds(self):
        stats = summarise([0.01, 0.02, 0.03])
        assert stats["median_ms"] == pytest.approx(20.0)
        assert stats["mean_ms"] == pytest.approx(20.0)

    def test_throughput_matches_mean(self):
        stats = summarise([0.1, 0.1])
        assert stats["items_per_s"] == pytest.approx(10.0)

    def test_p95_never_below_median(self):
        stats = summarise([0.001, 0.002, 0.1])
        assert stats["p95_ms"] >= stats["median_ms"]

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            summarise([])

    def test_zero_latency_does_not_divide_by_zero(self):
        stats = summarise([0.0])
        assert stats["items_per_s"] == float("inf")


class TestDefaults:
    def test_option_counts_cover_the_advertised_ceiling(self):
        assert 2 in DEFAULT_OPTION_COUNTS
        assert max(DEFAULT_OPTION_COUNTS) >= 256
