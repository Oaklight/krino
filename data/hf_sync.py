"""Fetch pinned data files from the Krino HuggingFace dataset.

The code repo does not carry data.  ``data/hf.lock`` pins the HF revision and
maps every remote file to its local path, byte size and SHA-256.  This module
downloads only the files named in the lock, verifies their hashes, and copies
them into the local paths the loaders already use.

Usage:
    python -m data.hf_sync --check          # verify local files only
    python -m data.hf_sync                  # download and verify everything
    python -m data.hf_sync --only 'derived/abstention/*'
    python -m data.hf_sync --list
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_LOCK = Path(__file__).with_name("hf.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    """Read and validate a lock file."""
    lock = json.loads(path.read_text())
    for field in ("repo", "repo_type", "revision", "files"):
        if not lock.get(field):
            raise ValueError(f"lock is missing {field!r}")
    files = lock["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("lock has no files")
    local_paths: set[str] = set()
    for remote, spec in files.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{remote}: spec must be an object")
        for field in ("local", "sha256", "bytes"):
            if field not in spec:
                raise ValueError(f"{remote}: missing {field!r}")
        local = str(spec["local"])
        if local in local_paths:
            raise ValueError(f"duplicate local path {local!r}")
        local_paths.add(local)
        if len(str(spec["sha256"])) != 64:
            raise ValueError(f"{remote}: sha256 must be 64 hex characters")
    return lock


def select_files(lock: dict[str, Any], patterns: list[str] | None) -> dict[str, dict]:
    files = lock["files"]
    if not patterns:
        return dict(files)
    chosen = {
        remote: spec
        for remote, spec in files.items()
        if any(fnmatch.fnmatch(remote, pattern) for pattern in patterns)
    }
    if not chosen:
        raise ValueError(f"no lock entries match {patterns}")
    return chosen


def verify_local(lock: dict[str, Any], dest: Path, patterns: list[str] | None = None) -> dict:
    """Verify local copies without touching the network."""
    report = {"ok": 0, "missing": 0, "mismatched": 0, "failures": []}
    for remote, spec in select_files(lock, patterns).items():
        path = dest / spec["local"]
        if not path.exists():
            report["missing"] += 1
            report["failures"].append({"remote": remote, "reason": "missing"})
            continue
        if path.stat().st_size != int(spec["bytes"]):
            report["mismatched"] += 1
            report["failures"].append({"remote": remote, "reason": "size"})
            continue
        if sha256(path) != spec["sha256"]:
            report["mismatched"] += 1
            report["failures"].append({"remote": remote, "reason": "sha256"})
            continue
        report["ok"] += 1
    return report


def sync(
    lock: dict[str, Any],
    dest: Path,
    patterns: list[str] | None = None,
    downloader: Callable[..., str] | None = None,
) -> dict:
    """Download the selected files, verify them and copy to their local paths."""
    selected = select_files(lock, patterns)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    cache = Path(
        downloader(
            repo_id=lock["repo"],
            repo_type=lock["repo_type"],
            revision=lock["revision"],
            allow_patterns=list(selected),
        )
    )
    report = {"copied": 0, "verified": 0, "missing": 0, "mismatched": 0, "failures": []}
    for remote, spec in selected.items():
        source = cache / remote
        if not source.exists():
            report["missing"] += 1
            report["failures"].append({"remote": remote, "reason": "download missing"})
            continue
        if source.stat().st_size != int(spec["bytes"]):
            report["mismatched"] += 1
            report["failures"].append({"remote": remote, "reason": "size"})
            continue
        if sha256(source) != spec["sha256"]:
            report["mismatched"] += 1
            report["failures"].append({"remote": remote, "reason": "sha256"})
            continue
        target = dest / spec["local"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report["copied"] += 1
        if sha256(target) == spec["sha256"]:
            report["verified"] += 1
        else:
            report["mismatched"] += 1
            report["failures"].append({"remote": remote, "reason": "copy sha256"})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--dest", type=Path, default=Path.cwd())
    parser.add_argument("--only", action="append", default=None,
                        help="glob on the remote path; repeatable")
    parser.add_argument("--check", action="store_true",
                        help="verify local files without downloading")
    parser.add_argument("--list", action="store_true", help="print the lock entries")
    args = parser.parse_args()

    lock = load_lock(args.lock)
    if args.list:
        for remote, spec in select_files(lock, args.only).items():
            print(f"{spec['bytes']:>12}  {remote}  ->  {spec['local']}")
        return 0
    report = (
        verify_local(lock, args.dest, args.only)
        if args.check
        else sync(lock, args.dest, args.only)
    )
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
