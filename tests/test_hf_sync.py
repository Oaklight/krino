"""Tests for the HF data lock and sync helper (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.hf_sync import (
    load_lock,
    select_files,
    sha256,
    sync,
    verify_local,
)


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _lock(tmp_path: Path, files: dict) -> Path:
    lock = {
        "repo": "oaklight/krino-data",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "files": files,
    }
    path = tmp_path / "hf.lock"
    path.write_text(json.dumps(lock))
    return path


def test_load_lock_validates_fields_and_duplicates(tmp_path: Path):
    good = _lock(tmp_path, {
        "derived/a.jsonl": {"local": "data/a.jsonl", "sha256": "b" * 64, "bytes": 1},
    })
    assert load_lock(good)["repo"] == "oaklight/krino-data"

    bad = _lock(tmp_path, {"derived/a.jsonl": {"local": "data/a.jsonl", "sha256": "short", "bytes": 1}})
    with pytest.raises(ValueError, match="sha256"):
        load_lock(bad)

    duplicate = _lock(tmp_path, {
        "derived/a.jsonl": {"local": "data/a.jsonl", "sha256": "b" * 64, "bytes": 1},
        "derived/b.jsonl": {"local": "data/a.jsonl", "sha256": "c" * 64, "bytes": 1},
    })
    with pytest.raises(ValueError, match="duplicate"):
        load_lock(duplicate)


def test_select_files_uses_globs(tmp_path: Path):
    lock = load_lock(_lock(tmp_path, {
        "derived/abstention/20k/train.jsonl": {"local": "data/a", "sha256": "b" * 64, "bytes": 1},
        "derived/long_context/x.jsonl": {"local": "data/b", "sha256": "c" * 64, "bytes": 1},
    }))
    chosen = select_files(lock, ["derived/abstention/*"])
    assert list(chosen) == ["derived/abstention/20k/train.jsonl"]
    with pytest.raises(ValueError, match="no lock entries"):
        select_files(lock, ["nope/*"])


def test_verify_local_reports_ok_missing_and_mismatch(tmp_path: Path):
    content = b"hello"
    digest = _write(tmp_path / "local" / "a.jsonl", content)
    lock = load_lock(_lock(tmp_path, {
        "derived/a.jsonl": {"local": "local/a.jsonl", "sha256": digest, "bytes": len(content)},
        "derived/b.jsonl": {"local": "local/b.jsonl", "sha256": "b" * 64, "bytes": 3},
    }))
    report = verify_local(lock, tmp_path)
    assert report["ok"] == 1 and report["missing"] == 1

    (tmp_path / "local" / "a.jsonl").write_bytes(b"changed")
    report = verify_local(lock, tmp_path)
    assert report["mismatched"] == 1


def test_sync_copies_and_verifies_with_a_fake_downloader(tmp_path: Path):
    content = b"derived payload"
    digest = hashlib.sha256(content).hexdigest()
    cache = tmp_path / "cache"
    _write(cache / "derived" / "a.jsonl", content)
    lock = load_lock(_lock(tmp_path, {
        "derived/a.jsonl": {"local": "data/benchmarks/a.jsonl", "sha256": digest, "bytes": len(content)},
    }))

    calls = {}

    def fake_downloader(**kwargs):
        calls.update(kwargs)
        return str(cache)

    report = sync(lock, tmp_path, downloader=fake_downloader)
    assert report["copied"] == 1 and report["verified"] == 1
    assert (tmp_path / "data/benchmarks/a.jsonl").read_bytes() == content
    assert calls["repo_id"] == "oaklight/krino-data"
    assert calls["revision"] == "a" * 40
    assert calls["allow_patterns"] == ["derived/a.jsonl"]


def test_sync_rejects_a_hash_mismatch(tmp_path: Path):
    cache = tmp_path / "cache"
    _write(cache / "derived" / "a.jsonl", b"tampered")
    lock = load_lock(_lock(tmp_path, {
        "derived/a.jsonl": {"local": "data/a.jsonl", "sha256": "0" * 64, "bytes": 8},
    }))
    report = sync(lock, tmp_path, downloader=lambda **kwargs: str(cache))
    assert report["mismatched"] == 1
    assert report["copied"] == 0
    assert not (tmp_path / "data/a.jsonl").exists()


def test_sha256_matches_hashlib(tmp_path: Path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert sha256(path) == hashlib.sha256(b"abc").hexdigest()
