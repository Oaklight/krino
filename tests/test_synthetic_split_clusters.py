"""Tests for cluster-aware synthetic splitting (issue #120).

Imports nothing from torch, so it runs on a bare interpreter (and in CI).
"""

from __future__ import annotations

import json

from data.synthetic_split import (
    CACHE_NAME,
    CACHE_VERSION,
    assign_splits,
    cluster_families,
    corpus_fingerprint,
    domain_splits,
    family_state_texts,
    split_from_key,
)

BASE_STATE = (
    "The team is preparing a release and the build is one day from the freeze. "
    "Two failing tests block the branch, and the release manager asks whether "
    "to ship with known failures or delay the release."
)
# Differs by one word, so char 5-gram Jaccard stays well above 0.7.
NEAR_DUP_STATE = BASE_STATE.replace("one day", "two days")
UNRELATED_STATE = (
    "A patient presents with a persistent cough for three weeks after a "
    "respiratory infection. The physician must decide between watchful waiting "
    "and a chest radiograph."
)


def _family(state: str) -> dict:
    return {"state": state}


class TestFamilyStateTexts:
    def test_includes_all_three_states(self):
        family = {"state": "base"}
        variants = {
            "counterfactual": {"state": "cf state"},
            "paraphrase": {"state_paraphrase": "para state"},
        }
        assert family_state_texts(family, variants) == ["base", "cf state", "para state"]

    def test_skips_missing_and_empty_states(self):
        assert family_state_texts({"state": "base"}, {}) == ["base"]
        assert family_state_texts({"state": "base"}, None) == ["base"]
        assert family_state_texts({}, None) == []

    def test_deduplicates_and_normalises(self):
        family = {"state": "a  b\n c "}
        variants = {"paraphrase": {"state_paraphrase": "a b c"}}
        assert family_state_texts(family, variants) == ["a b c"]


class TestClusterFamilies:
    def test_near_duplicates_share_a_cluster(self):
        clusters = cluster_families(
            [[BASE_STATE], [NEAR_DUP_STATE], [UNRELATED_STATE]]
        )
        assert clusters[0] == clusters[1]
        assert clusters[2] != clusters[0]

    def test_unrelated_states_stay_singletons(self):
        clusters = cluster_families([[BASE_STATE], [UNRELATED_STATE]])
        assert clusters == [0, 1]

    def test_cluster_label_is_lowest_index(self):
        clusters = cluster_families([[UNRELATED_STATE], [NEAR_DUP_STATE], [BASE_STATE]])
        assert clusters == [0, 1, 1]

    def test_clustering_is_transitive(self):
        """A~B~C~D puts A and D together even though A and D fall below 0.7.

        Each step replaces one phrase everywhere in a six-copy paragraph, so
        consecutive states stay at 0.82-0.88 Jaccard while the ends reach
        ~0.62. A chain that only held by direct comparison would leave the
        ends in separate clusters.
        """
        segment = (
            "The team is preparing a release and the build is one day from the "
            "freeze. Two failing tests block the branch, and the release manager "
            "asks whether to ship with known failures or delay the release. "
        )
        base = segment * 6
        steps = [
            ("one day", "two full days"),
            ("known failures", "unrelated defects"),
            ("release manager", "duty officer"),
        ]
        states = [base]
        for old, new in steps:
            states.append(states[-1].replace(old, new))

        clusters = cluster_families([[state] for state in states])
        assert len(set(clusters)) == 1

    def test_match_through_a_variant_state(self):
        """A family is linked when any of its states matches, not just the base."""
        left = [_family(BASE_STATE), _family(UNRELATED_STATE)]
        left_states = [family_state_texts(f) for f in left]
        left_states[1].append(NEAR_DUP_STATE)
        clusters = cluster_families(left_states)
        assert clusters[0] == clusters[1]

    def test_empty_input(self):
        assert cluster_families([]) == []


class TestAssignSplits:
    def test_cluster_members_share_a_split(self):
        splits = assign_splits("domain", [0, 0, 2])
        assert splits[0] == splits[1]
        assert splits[2] in ("train", "test")

    def test_singleton_keeps_family_level_split(self):
        """A family on its own gets the split the family-level scheme gives it."""
        from data.pipeline import _synthetic_family_split

        splits = assign_splits("medical_triage", [0, 1, 2])
        assert splits == [_synthetic_family_split("medical_triage", i) for i in range(3)]

    def test_split_from_key_matches_family_scheme(self):
        from data.pipeline import _synthetic_family_split

        for idx in range(20):
            assert split_from_key(f"medical_triage-{idx}") == (
                _synthetic_family_split("medical_triage", idx)
            )


class TestDomainSplits:
    def test_aligns_with_families_and_links_duplicates(self):
        families = [_family(BASE_STATE), _family(NEAR_DUP_STATE), _family(UNRELATED_STATE)]
        splits = domain_splits("d", families)
        assert len(splits) == 3
        assert splits[0] == splits[1]

    def test_family_idx_is_used_for_the_key(self):
        """Split keys come from ``_family_idx`` so renumbering is irrelevant."""
        families = [_family(BASE_STATE), _family(UNRELATED_STATE)]
        splits = domain_splits("d", families, family_idxs=[7, 9])
        assert splits[0] == split_from_key("d-7")
        assert splits[1] == split_from_key("d-9")

    def test_empty_domain(self):
        assert domain_splits("d", []) == []

    def test_cache_round_trip_avoids_reclustering(self, tmp_path, monkeypatch):
        cache = tmp_path / CACHE_NAME
        families = [_family(BASE_STATE), _family(UNRELATED_STATE)]
        first = domain_splits(
            "d", families, content_key="abc", cache_path=cache
        )
        assert cache.exists()

        import data.synthetic_split as module

        def boom(*args, **kwargs):
            raise AssertionError("clustering should not rerun on a cache hit")

        monkeypatch.setattr(module, "cluster_families", boom)
        second = domain_splits("d", families, content_key="abc", cache_path=cache)
        assert second == first

    def test_cache_invalidated_by_content_key(self, tmp_path, monkeypatch):
        cache = tmp_path / CACHE_NAME
        families = [_family(BASE_STATE)]
        domain_splits("d", families, content_key="abc", cache_path=cache)

        import data.synthetic_split as module

        calls = []
        original = module.cluster_families

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "cluster_families", spy)
        domain_splits("d", families, content_key="different", cache_path=cache)
        assert calls, "a changed corpus must invalidate the cache"

    def test_cache_invalidated_by_threshold(self, tmp_path, monkeypatch):
        cache = tmp_path / CACHE_NAME
        families = [_family(BASE_STATE)]
        domain_splits("d", families, content_key="abc", cache_path=cache)

        import data.synthetic_split as module

        calls = []
        original = module.cluster_families

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "cluster_families", spy)
        domain_splits("d", families, threshold=0.9, content_key="abc", cache_path=cache)
        assert calls, "a changed threshold must invalidate the cache"

    def test_unwritable_cache_does_not_fail(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        splits = domain_splits(
            "d", [_family(BASE_STATE)], content_key="abc", cache_path=blocker / CACHE_NAME
        )
        assert splits == [split_from_key("d-0")]

    def test_cache_file_is_valid_json(self, tmp_path):
        cache = tmp_path / CACHE_NAME
        domain_splits("d", [_family(BASE_STATE)], content_key="abc", cache_path=cache)
        payload = json.loads(cache.read_text())
        assert payload["version"] == CACHE_VERSION
        assert "d" in payload["domains"]


class TestCorpusFingerprint:
    def test_changes_with_content(self, tmp_path):
        path = tmp_path / "d_families.jsonl"
        path.write_text("{}\n")
        first = corpus_fingerprint([path])
        path.write_text('{"state": "changed"}\n')
        assert corpus_fingerprint([path]) != first

    def test_stable_for_identical_content(self, tmp_path):
        left = tmp_path / "a" / "d_families.jsonl"
        right = tmp_path / "b" / "d_families.jsonl"
        left.parent.mkdir()
        right.parent.mkdir()
        left.write_text("{}\n")
        right.write_text("{}\n")
        assert corpus_fingerprint([left]) == corpus_fingerprint([right])

    def test_missing_file_disables_cache(self, tmp_path):
        assert corpus_fingerprint([tmp_path / "absent.jsonl"]) == ""
