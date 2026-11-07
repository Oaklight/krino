"""Tests for the canonical builder's dedup and exclusion.

The builder is a script, so each test assembles a minimal corpus tree under
``tmp_path``: every required file exists, and only the files under test carry
rows.  Rows declare their provenance tier so the builder's label and
provenance contracts cannot mask the filter under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.build_canonical_corpus import CANONICAL_FILES, main

REPO_ROOT = Path(__file__).resolve().parents[1]
INCLUDED = "sst2"


def _row(item_id: str, source: str, label: bool = True) -> dict:
    return {
        "id": item_id,
        "state": "the evidence",
        "question": {"type": "noul", "instructions": "is it true?"},
        "label": label,
        "source": source,
        "split": "train",
        "provenance": "gold",
    }


def _manifest(**groups: list[str]) -> dict:
    return {
        "groups": {name: {"included_in": builds} for name, builds in groups.items()}
    }


def _corpus(tmp_path: Path, rows: dict[str, list[dict]], manifest: dict) -> Path:
    root = tmp_path / "benchmarks"
    root.mkdir()
    for _, name in CANONICAL_FILES:
        path = root / "typed_decisions.jsonl" if name is None else root / name
        path.write_text("", encoding="utf-8")
    (root / "synthetic").mkdir()
    (root / "synthetic" / "unit_families.jsonl").write_text("", encoding="utf-8")
    for name, records in rows.items():
        (root / name).write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _build(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    build: str = "exclude_restrictive",
) -> tuple[dict, Path]:
    out = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_canonical_corpus.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--build",
            build,
            "--root",
            str(root),
            "--repo-root",
            str(REPO_ROOT),
            "--out",
            str(out),
            "--json",
            str(report),
        ],
    )
    assert main() == 0
    return json.loads(report.read_text(encoding="utf-8")), out


def _written(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


@pytest.mark.feature_guard("KRINO_DISABLE_CANONICAL_DEDUP")
def test_a_repeated_id_across_corpora_is_kept_once(tmp_path, monkeypatch):
    rows = {
        "label_target_30k_jev.jsonl": [_row("dup", INCLUDED, True)],
        "residual_train_20000_jev.jsonl": [_row("dup", INCLUDED, False)],
    }
    root = _corpus(tmp_path, rows, _manifest(**{INCLUDED: ["exclude_restrictive"]}))
    report, out = _build(tmp_path, root, monkeypatch)
    assert report["total"] == 1
    assert [row["id"] for row in _written(out)] == ["dup"]


@pytest.mark.feature_guard("KRINO_DISABLE_CANONICAL_DEDUP")
def test_a_repeated_id_in_one_corpus_is_disambiguated(tmp_path, monkeypatch):
    rows = {
        "label_target_30k_jev.jsonl": [
            _row("dup", INCLUDED, True),
            _row("dup", INCLUDED, False),
        ],
    }
    root = _corpus(tmp_path, rows, _manifest(**{INCLUDED: ["exclude_restrictive"]}))
    report, out = _build(tmp_path, root, monkeypatch)
    written = _written(out)
    assert [row["id"] for row in written] == ["dup-q1", "dup-q2"]
    assert [row["group"] for row in written] == ["dup", "dup"]
    assert report["total"] == 2


@pytest.mark.feature_guard("KRINO_DISABLE_CANONICAL_LICENCE_EXCLUSION")
def test_a_group_outside_the_build_is_dropped(tmp_path, monkeypatch):
    rows = {
        "label_target_30k_jev.jsonl": [
            _row("keep", INCLUDED),
            _row("drop", "unit_excluded"),
        ],
    }
    manifest = _manifest(
        **{INCLUDED: ["exclude_restrictive"], "unit_excluded": ["full"]}
    )
    root = _corpus(tmp_path, rows, manifest)
    report, out = _build(tmp_path, root, monkeypatch)
    assert report["total"] == 1
    assert report["dropped_groups"] == {"unit_excluded": 1}
    assert [row["id"] for row in _written(out)] == ["keep"]


@pytest.mark.feature_guard("KRINO_DISABLE_CANONICAL_UNKNOWN_GROUP_GUARD")
def test_an_unknown_group_fails_closed(tmp_path, monkeypatch):
    rows = {"label_target_30k_jev.jsonl": [_row("unknown", "unit_unknown")]}
    root = _corpus(tmp_path, rows, _manifest(**{INCLUDED: ["full"]}))
    with pytest.raises(SystemExit, match="fail closed"):
        _build(tmp_path, root, monkeypatch, build="full")
