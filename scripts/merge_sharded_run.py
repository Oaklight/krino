#!/usr/bin/env python3
"""Merge a sharded ``eval_logit`` run into one result file.

A run is split by benchmark, one shard per GPU, because the benchmarks are
independent: ``LogitScorer`` carries no state between datasets, so the split
cannot change a score.  The shards are not interchangeable files, though: each
one's ``_provenance.command`` names only its own datasets, and a naive copy-merge
would leave the merged row claiming it scored a fifth of the benchmark set.

This script is what makes the merge acceptable:

- every shard's ``_provenance`` must be identical to the first apart from the
  two fields that legitimately differ between processes, ``command`` and
  ``date_utc``, so no shard ran different settings, dtype, revision or code;
  the merged ``date_utc`` is the earliest shard's start and each shard's own
  start is recorded under ``shards``;
- the merged ``command`` is rebuilt as the full invocation, listing every
  dataset in the order given on the command line;
- ``_provenance.shards`` records each shard's index, its benchmark names and its
  status - names only, no paths, so the merged file needs no extra redaction;
- it refuses duplicate benchmarks, a missing benchmark, a benchmark whose
  ``items`` disagrees with its per-type ``n``, and any shard whose status file
  records a non-zero exit.

Usage:
    python scripts/merge_sharded_run.py \
        --out /tmp/qwen35_all19.json \
        --datasets /data/benchmarks/openjev/authored144.jsonl /data/benchmarks/agnews.jsonl ... \
        --status-dir /tmp/krino-out/qwen35 \
        /tmp/krino-out/qwen35/part_0.json ... /tmp/krino-out/qwen35/part_7.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

DATASET_FLAG = " --datasets "
#: Fields that legitimately differ between shards: each process writes its own
#: command and start time.  Everything else has to match or the shards were not
#: the same run.
PER_SHARD_FIELDS = ("command", "date_utc")


def benchmark_names(payload: dict[str, Any]) -> list[str]:
    return [key for key in payload if not key.startswith("_")]


def shard_invariant_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in provenance.items() if key not in PER_SHARD_FIELDS
    }


def full_command(parts: Sequence[dict[str, Any]], datasets: Sequence[str]) -> str:
    """Rebuild the invocation with every dataset, from any shard's prefix."""
    command = parts[0]["_provenance"].get("command", "")
    if DATASET_FLAG not in command:
        raise ValueError("a shard's command has no `--datasets` section to extend")
    prefix = command.split(DATASET_FLAG, 1)[0]
    return prefix + DATASET_FLAG + " ".join(datasets)


def check_items(payload: dict[str, Any], name: str) -> None:
    """Refuse a benchmark whose item count disagrees with its scored count."""
    block = payload[name]
    items = block.get("items")
    if items is None:
        return
    for key in ("noul", "choice"):
        accuracy = block.get(key, {}).get("accuracy", {})
        if accuracy.get("n") is not None and accuracy["n"] != items:
            raise ValueError(
                f"{name}: items={items} but {key} accuracy n={accuracy['n']}"
            )


def read_status(status_dir: Path | None, index: int) -> str:
    if status_dir is None:
        return "unknown"
    path = status_dir / f"part_{index}.status"
    if not path.exists():
        return "missing"
    text = path.read_text().strip()
    if "exit=0" in text:
        return "ok"
    return f"failed: {text}"


def merge(
    parts: Sequence[dict[str, Any]],
    datasets: Sequence[str],
    status_dir: Path | None = None,
) -> dict[str, Any]:
    if not parts:
        raise ValueError("no shards")
    base = shard_invariant_provenance(parts[0]["_provenance"])
    merged: dict[str, Any] = {"_provenance": dict(base)}
    shards = []
    seen: set[str] = set()
    dates = []
    for index, part in enumerate(parts):
        provenance = part["_provenance"]
        dates.append(provenance.get("date_utc"))
        if shard_invariant_provenance(provenance) != base:
            differing = [
                key
                for key in base
                if provenance.get(key) != base[key]
            ] + [key for key in provenance if key not in base]
            raise ValueError(
                f"shard {index} provenance differs from shard 0 outside "
                f"{PER_SHARD_FIELDS}: {sorted(set(differing))}"
            )
        status = read_status(status_dir, index)
        names = benchmark_names(part)
        for name in names:
            if name in seen:
                raise ValueError(f"benchmark {name!r} appears in more than one shard")
            seen.add(name)
            check_items(part, name)
            merged[name] = part[name]
        shards.append(
            {
                "index": index,
                "benchmarks": names,
                "status": status,
                "started_utc": provenance.get("date_utc"),
            }
        )
        if status.startswith("failed") or status == "missing":
            raise ValueError(f"shard {index} is not ok: {status}")
    expected = {Path(dataset).stem for dataset in datasets}
    if seen != expected:
        raise ValueError(
            f"benchmarks do not match the dataset list: missing {sorted(expected - seen)}, "
            f"extra {sorted(seen - expected)}"
        )
    present = [date for date in dates if isinstance(date, str)]
    if present:
        merged["_provenance"]["date_utc"] = min(present)
    merged["_provenance"]["command"] = full_command(parts, datasets)
    merged["_provenance"]["shards"] = shards
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="every dataset of the run, in the order the merged command should list them",
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="directory holding part_<n>.status files written by the shard script",
    )
    args = parser.parse_args()

    parts = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    merged = merge(parts, args.datasets, args.status_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    names = benchmark_names(merged)
    print(f"{len(parts)} shards -> {len(names)} benchmarks in {args.out}")
    for shard in merged["_provenance"]["shards"]:
        print(f"  shard {shard['index']}: {shard['status']} {shard['benchmarks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
