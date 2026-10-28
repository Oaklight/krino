"""Tests for the empty-source guard.

A configured source that loads nothing must stop the run: a fresh worktree
without the prepared benchmark files trained on 10k of 182k items with one
stderr warning per source as the only signal. Torch-free, so it runs in CI.
"""

from __future__ import annotations

from data.sources import missing_sources


class TestMissingSources:
    def test_empty_source_is_reported(self):
        counts = {"sst2": (0, 0), "banking77": (5000, 500)}
        assert missing_sources(counts, {}) == ["sst2"]

    def test_allow_missing_opts_out(self):
        counts = {"sst2": (0, 0)}
        cfg = {"sst2": {"allow_missing": True}}
        assert missing_sources(counts, cfg) == []

    def test_eval_only_source_is_not_missing(self):
        """weight: 0 sources legitimately yield no train items."""
        counts = {"authored144": (0, 144)}
        cfg = {"authored144": {"weight": 0}}
        assert missing_sources(counts, cfg) == []

    def test_train_only_source_is_not_missing(self):
        assert missing_sources({"src": (10, 0)}, {}) == []

    def test_none_config_entry_is_tolerated(self):
        counts = {"src": (0, 0)}
        assert missing_sources(counts, {"src": None}) == ["src"]

    def test_multiple_missing_sources_all_listed(self):
        counts = {"a": (0, 0), "b": (0, 0), "c": (1, 1)}
        assert missing_sources(counts, {}) == ["a", "b"]

    def test_order_follows_the_counts_mapping(self):
        counts = {"z": (0, 0), "a": (0, 0)}
        assert missing_sources(counts, {}) == ["z", "a"]

    def test_empty_counts(self):
        assert missing_sources({}, {}) == []
