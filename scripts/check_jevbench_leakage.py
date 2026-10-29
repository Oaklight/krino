#!/usr/bin/env python3
"""Ask whether the JevBench items we report on could have been trained on.

Clause 4 of the goal quotes JevBench-public next to Jev's reference, and the
goal's premise is that the number is held-out. JevBench is not a training source
in any config, so the only way it is not held out is if its items duplicate items
from the benchmark corpus we do train on. This measures that, from the files.

Two questions, because they fail differently:

* **Exact.** Does a probe item appear in the training split with the same state
  and the same options? Then either the gold agrees -- a duplicate the model was
  trained on -- or it does not, which is a label conflict rather than leakage.
  Both matter, for different reasons.
* **Near-duplicate.** Does a probe state share most of its eight-word shingles
  with a training state? A reworded copy is not caught by exact matching and is
  the case a corpus assembled from public sources would actually present. The
  shingle containment is reported as a distribution rather than a single yes, and
  the pairs above the reporting threshold are named.

JevBench is authored rather than sampled from the benchmarks it resembles, so
the expected answer is zero on both. That is worth establishing rather than
assuming: it is the only claim in clause 4 that a reader cannot re-derive from
the repository.

Usage:
    python scripts/check_jevbench_leakage.py \
        --probe data/benchmarks/jevbench/easy.jsonl \
        --probe data/benchmarks/jevbench/hard.jsonl \
        --probe data/benchmarks/jevbench/original.jsonl \
        --data-dir data/benchmarks --json model/experiments/jevbench_leakage.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Files that are not training corpora even though they sit beside them.
NOT_TRAINING = {"jevbench"}
SHINGLE_WORDS = 8
REPORT_THRESHOLD = 0.5

_WHITESPACE = re.compile(r"\s+")


class TrainingEntry(NamedTuple):
    source: str
    item_id: str
    question: str
    gold: str


def normalise(text: Any) -> str:
    """Collapse whitespace and case so formatting cannot hide a duplicate."""
    return _WHITESPACE.sub(" ", str(text)).strip().lower()


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def shingle_hashes(text: str, size: int = SHINGLE_WORDS) -> set[int]:
    """Hash every run of ``size`` words in ``text``.

    Shingles shorter than ``size`` words (a very short state) fall back to the
    whole text, so a short item still has one comparable feature rather than
    none and silently scoring containment 0.
    """
    words = normalise(text).split()
    if not words:
        return set()
    if len(words) < size:
        return {_digest(" ".join(words))}
    return {
        _digest(" ".join(words[start : start + size]))
        for start in range(len(words) - size + 1)
    }


def canonical_question(question: dict[str, Any] | None) -> str:
    """Serialise the parts of a question that decide what is being asked.

    Criteria appear as a mapping or as a list and their order differs between
    sources without changing the question, so both shapes are sorted before
    hashing. Instructions are included because two sources can reuse a state
    under different questions.
    """
    question = question or {}
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        criteria_key: Any = sorted((normalise(k), normalise(v)) for k, v in criteria.items())
    elif isinstance(criteria, list):
        criteria_key = sorted(normalise(value) for value in criteria)
    else:
        criteria_key = None
    return json.dumps(
        {
            "type": question.get("type"),
            "instructions": normalise(question.get("instructions", "")),
            "criteria": criteria_key,
        },
        sort_keys=True,
    )


def read_items(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_probes(paths: list[Path]) -> list[dict[str, Any]]:
    """Load probe items, keeping the public-dataset shape (``expected``)."""
    probes: list[dict[str, Any]] = []
    for path in paths:
        for row in read_items(path):
            probes.append({"tier": path.stem, "row": row, "file": path.name})
    return probes


def load_training_index(data_dir: Path) -> tuple[dict[str, list[TrainingEntry]], set[int], int]:
    """Index the training split of every corpus file by normalised state.

    Args:
        data_dir: Directory holding ``<source>.jsonl`` files.

    Returns:
        ``(state_index, shingle_set, n_train)`` where ``state_index`` maps a
        normalised state to its training entries and ``shingle_set`` holds every
        eight-word shingle hash in the training split.
    """
    state_index: dict[str, list[TrainingEntry]] = defaultdict(list)
    shingle_set: set[int] = set()
    n_train = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        source = path.stem
        if source in NOT_TRAINING:
            continue
        for row in read_items(path):
            if row.get("split") != "train":
                continue
            n_train += 1
            state = normalise(row.get("state", ""))
            state_index[state].append(
                TrainingEntry(
                    source=source,
                    item_id=str(row.get("id")),
                    question=canonical_question(row.get("question")),
                    gold=normalise(row.get("label", "")),
                )
            )
            shingle_set |= shingle_hashes(row.get("state", ""))
    return state_index, shingle_set, n_train


def gold_of(row: dict[str, Any]) -> str:
    return normalise(row.get("expected", row.get("label", "")))


def classify_exact(
    entry: TrainingEntry, question: str, gold: str
) -> str:
    """Label one exact state match.

    Returns ``"identical"`` (same question and gold), ``"label_conflict"`` (same
    question, different gold) or ``"state_repeat"`` (the state recurs under a
    different question, which on its own is not leakage).
    """
    if entry.question != question:
        return "state_repeat"
    return "identical" if entry.gold == gold else "label_conflict"


def audit(probes: list[dict[str, Any]], data_dir: Path, threshold: float) -> dict[str, Any]:
    state_index, shingle_set, n_train = load_training_index(data_dir)
    per_tier: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "probes": 0,
            "state_repeat": 0,
            "identical": 0,
            "label_conflict": 0,
            "near": [],
        }
    )
    containments: list[float] = []

    for probe in probes:
        tier = per_tier[probe["tier"]]
        tier["probes"] += 1
        row = probe["row"]
        state = normalise(row.get("state", ""))
        question = canonical_question(row.get("question"))
        gold = gold_of(row)

        kinds = {classify_exact(entry, question, gold) for entry in state_index.get(state, [])}
        for kind in kinds:
            tier[kind] += 1

        hashes = shingle_hashes(row.get("state", ""))
        if hashes:
            shared = len(hashes & shingle_set)
            containment = shared / len(hashes)
            containments.append(containment)
            if containment >= threshold:
                tier["near"].append(
                    {
                        "id": row.get("id"),
                        "file": probe["file"],
                        "containment": round(containment, 4),
                        "shingles": len(hashes),
                        "shared": shared,
                        "exact_state": bool(kinds),
                    }
                )

    containments.sort()
    return {
        "training_items": n_train,
        "training_shingles": len(shingle_set),
        "probes": len(probes),
        "threshold": threshold,
        "max_containment": round(containments[-1], 4) if containments else 0.0,
        "p99_containment": (
            round(containments[min(len(containments) - 1, int(0.99 * len(containments)))], 4)
            if containments
            else 0.0
        ),
        "probes_at_or_above_threshold": sum(1 for value in containments if value >= threshold),
        "by_tier": {
            tier: {key: value for key, value in data.items() if key != "near"}
            | {"near": sorted(data["near"], key=lambda row: -row["containment"])}
            for tier, data in sorted(per_tier.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--threshold", type=float, default=REPORT_THRESHOLD)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    probes = load_probes(args.probe)
    print(f"{len(probes)} probe items from {len(args.probe)} files", flush=True)
    report = audit(probes, args.data_dir, args.threshold)

    print(
        f"training side: {report['training_items']:,} items, "
        f"{report['training_shingles']:,} distinct shingles",
        flush=True,
    )
    print(
        f"shingle containment: max {report['max_containment']:.3f}, "
        f"p99 {report['p99_containment']:.3f}, "
        f"{report['probes_at_or_above_threshold']} at or above {report['threshold']}",
        flush=True,
    )
    for tier, data in report["by_tier"].items():
        print(
            f"  {tier:<10} n={data['probes']:>4} state_repeat={data['state_repeat']:>3} "
            f"identical={data['identical']:>3} label_conflict={data['label_conflict']:>3} "
            f"near={len(data['near']):>3}",
            flush=True,
        )
        for row in data["near"][:5]:
            print(
                f"      {row['id']} containment={row['containment']:.3f} "
                f"({row['shared']}/{row['shingles']} shingles)",
                flush=True,
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
