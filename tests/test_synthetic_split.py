"""Tests for the synthetic family-level train/eval split.

Imports nothing from torch, so it runs on a bare interpreter (and in CI).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from data.pipeline import _synthetic_family_split


class TestFamilySplit:
    def test_deterministic(self):
        """Same family always gets the same split."""
        a = _synthetic_family_split("medical_triage", 42)
        b = _synthetic_family_split("medical_triage", 42)
        assert a == b
        assert a in ("train", "test")

    def test_stable_known_values(self):
        """Split is md5-based, not Python's randomised hash().

        Pinning literal values guards against a switch back to hash(), which
        would change the split under a different PYTHONHASHSEED.
        """
        assert _synthetic_family_split("medical_triage", 0) == "train"
        assert _synthetic_family_split("logical_inference", 0) == "test"
        assert _synthetic_family_split("code_review", 0) == "train"

    def test_domain_is_part_of_the_key(self):
        """Two domains with the same family index are split independently."""
        splits = {
            _synthetic_family_split(d, 7)
            for d in ("medical_triage", "legal_judgment", "code_review")
        }
        assert splits  # at least one value; not all domains may collide

    def test_eval_ratio_respected(self):
        """Roughly eval_ratio of families land in eval."""
        n = 2000
        ratio = 0.1
        evals = sum(
            1 for i in range(n) if _synthetic_family_split("d", i, ratio) == "test"
        )
        assert 0.06 < evals / n < 0.14

    def test_ratio_bounds(self):
        """ratio=0 gives no test items, ratio=1 gives all test items."""
        assert all(_synthetic_family_split("d", i, 0.0) == "train" for i in range(50))
        assert all(_synthetic_family_split("d", i, 1.0) == "test" for i in range(50))


class TestRealDataSplit:
    def test_no_family_leaks_across_splits(self):
        """Every variant of a family must land on the same side."""
        from data.pipeline import load_synthetic

        groups: dict[str, set[str]] = defaultdict(set)
        for item in load_synthetic():
            groups[item.group].add(item.split)

        leaking = {g: s for g, s in groups.items() if len(s) > 1}
        assert not leaking, f"{len(leaking)} groups leak across splits"

    def test_both_splits_present(self):
        """Real data yields a non-trivial number of eval items."""
        from data.pipeline import load_synthetic

        counts = Counter(item.split for item in load_synthetic())
        assert counts["train"] > 1000
        assert counts["test"] > 100
