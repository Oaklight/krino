"""The leakage checker must measure the split the loader assigns.

These tests run the checker itself, not a re-emission of the loader's calls.
That distinction is the point: the checker previously rebuilt the clustering
without ``force_group``, so on domains whose ``_family_idx`` values repeat it
described a split nobody assigns -- 182 clusters for ``education_assessment``
where the loader produces 151 -- and every number in its table was the wrong
corpus's number. A test that calls the same helpers the loader calls cannot
catch that; a test that runs the script can.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from data.pipeline import _synthetic_family_split
from data.synthetic_split import domain_splits

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    """Import ``scripts/check_split_leakage.py`` by path (scripts/ is not a package)."""
    path = ROOT / "scripts" / "check_split_leakage.py"
    spec = importlib.util.spec_from_file_location("check_split_leakage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = (
    "The team is preparing a release and the build is one day from the freeze. "
    "Two failing tests block the branch, and the release manager asks whether to "
    "ship with known failures or delay the release."
)
OTHER = (
    "A patient presents with a persistent cough for three weeks after a "
    "respiratory infection. The physician must decide between watchful waiting "
    "and a chest radiograph."
)
# Near-duplicate of OTHER, above the 0.7 threshold on character 5-grams.
OTHER_NEAR = OTHER.replace("three weeks", "four weeks")
UNRELATED = (
    "A routing table entry points at a stale next hop after a BGP withdrawal, "
    "and the operator must withdraw the route or wait for the timers to expire."
)

DOMAIN = "education_assessment"


def _splits_differing_idxs(domain: str) -> tuple[int, int]:
    """One ``_family_idx`` of each split, so a re-keyed family really moves."""
    train = next(i for i in range(64) if _synthetic_family_split(domain, i) == "train")
    test = next(i for i in range(64) if _synthetic_family_split(domain, i) == "test")
    return train, test


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """One domain where ``force_group`` changes the answer.

    Family ``UNRELATED`` and family ``OTHER`` share ``_family_idx`` = the test
    split's index.  ``OTHER_NEAR`` has the train split's index and duplicates
    ``OTHER``.  Without ``force_group`` the duplicate pair clusters on its own
    and ``UNRELATED`` keeps its own split; *with* it, the forced union pulls
    ``UNRELATED`` into a cluster keyed on the lowest index, so ``UNRELATED``
    changes split.
    """
    checker = _load_checker()
    synth_dir = tmp_path / "synthetic"
    synth_dir.mkdir(parents=True)
    train_idx, test_idx = _splits_differing_idxs(DOMAIN)

    families = [
        {"_family_idx": test_idx, "state": UNRELATED},
        {"_family_idx": test_idx, "state": OTHER},
        {"_family_idx": train_idx, "state": OTHER_NEAR},
    ]
    (synth_dir / f"{DOMAIN}_families.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in families)
    )
    (synth_dir / f"{DOMAIN}_variants.jsonl").write_text("")

    monkeypatch.setattr(checker, "DATA_DIR", tmp_path)
    return checker, synth_dir, families, (train_idx, test_idx)


def _run(checker) -> None:
    original = sys.argv
    sys.argv = ["check_split_leakage.py"]
    try:
        checker.main()
    finally:
        sys.argv = original


def test_fixture_exposes_the_missing_argument(corpus):
    """Guard: without ``force_group`` this corpus gives a different clustering."""
    from data.synthetic_split import cluster_families, family_state_texts

    _, _, families, _ = corpus
    states = [family_state_texts(fam, None) for fam in families]
    idxs = [fam["_family_idx"] for fam in families]
    forced = cluster_families(states, threshold=0.7, force_group=idxs)
    unforced = cluster_families(states, threshold=0.7)
    assert len(set(forced)) == 1, "force_group must merge all three families"
    assert unforced[0] != unforced[1], "without it, UNRELATED stays on its own"


def test_moved_count_matches_the_loader(corpus, capsys):
    """The headline moved count is what ``domain_splits`` assigns, not a rebuild."""
    checker, _, families, _ = corpus
    family_idxs = [fam["_family_idx"] for fam in families]
    splits = domain_splits(DOMAIN, families, family_idxs=family_idxs, threshold=0.7)
    expected = sum(
        1
        for split, idx in zip(splits, family_idxs)
        if split != _synthetic_family_split(DOMAIN, idx, 0.1)
    )
    assert expected >= 1, "the fixture must move at least one family"

    _run(checker)
    out = capsys.readouterr().out
    assert f"{len(families)} families, {expected} moved by clustering" in out
    assert f"moved={expected:>3}" in out
    assert "clusters=   1" in out


def test_no_positional_index_warning(corpus, capsys):
    """The warning described a bug that no longer exists and must not fire."""
    checker, _, _, _ = corpus
    _run(checker)
    assert "_family_idx is not positional" not in capsys.readouterr().err
