#!/usr/bin/env python3
"""Measure state leakage across the benchmark train/eval splits.

The synthetic corpus has its own checker (:mod:`scripts.check_split_leakage`),
which clusters near-duplicate states because the split is assigned per cluster.
Benchmark sources keep the split their files carry, so the question there is
narrower and answerable exactly: how many eval items also appear in train?

Three counts are reported per source, because they mean different things:

* **state repeats** — an eval item whose state also occurs in train, with a
  different question or label. In PIQA the goal sentence legitimately recurs
  with different option pairs, so this is mostly benign and is reported for
  context rather than as a failure.
* **identical items** — the same state, the same options and the same gold, on
  both sides. This is a duplicate the model was trained on; there is no reading
  of it other than leakage.
* **label conflicts** — the same state and options carrying different golds on
  the two sides. Not leakage, but a sign the label is unreliable, and worth
  knowing before a per-source accuracy is quoted.

Matches are exact after whitespace and case normalisation. Near-duplicate
leakage is not covered here; on benchmark sources it would need the same LSH
clustering the synthetic checker uses, over a corpus an order of magnitude
larger.

Usage:
    python scripts/check_benchmark_leakage.py
    python scripts/check_benchmark_leakage.py --data-dir data/benchmarks --json out.json
    python scripts/check_benchmark_leakage.py --max-identical-rate 0.005
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pipeline import DATA_DIR  # noqa: E402


def _canonical(text: Any) -> str:
    """Normalise a text field so whitespace and case cannot hide a duplicate."""
    return " ".join(str(text).split()).lower()


def _canonical_question(question: dict[str, Any] | None) -> str:
    """Serialise the parts of a question that decide what the item asks.

    Criteria can be a mapping (label -> description) or a list of option texts,
    and the order differs between sources without changing the question, so both
    shapes are sorted before hashing.
    """
    criteria = (question or {}).get("criteria")
    if criteria is None:
        return ""
    if isinstance(criteria, dict):
        pairs = sorted((str(key), _canonical(value)) for key, value in criteria.items())
        return "\x1e".join(f"{key}={value}" for key, value in pairs)
    return "\x1e".join(sorted(_canonical(option) for option in criteria))


def _digest(*parts: str) -> str:
    """Hash the parts together; the separator cannot occur in canonical text."""
    return hashlib.md5("\x1f".join(parts).encode()).hexdigest()


def iter_split_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(state, question, label, split)`` for every typed row of a file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question") or {}
            if question.get("type") not in {"noul", "choice", "score"}:
                continue
            yield (
                _canonical(row.get("state", "")),
                _canonical_question(question),
                str(row.get("label")),
                str(row.get("split", "")),
            )


def audit_file(path: Path) -> dict[str, Any]:
    """Count cross-split repeats in one source file.

    Args:
        path: JSONL file whose rows carry a ``split`` field.

    Returns:
        A dict with ``n_train``, ``n_test``, ``state_repeats``, ``identical``,
        ``label_conflicts`` and ``identical_rate``. ``identical_rate`` is
        relative to the test rows, and is ``0.0`` when there are none.
    """
    train_states: set[str] = set()
    train_items: dict[str, str] = {}
    test_rows: list[tuple[str, str, str]] = []
    n_train = 0

    for state, question, label, split in iter_split_rows(path):
        state_key = _digest(state)
        item_key = _digest(state, question)
        if split == "train":
            n_train += 1
            train_states.add(state_key)
            # First label wins; a source that disagrees with itself inside train
            # is a data problem this script does not claim to detect.
            train_items.setdefault(item_key, label)
        elif split == "test":
            test_rows.append((state_key, item_key, label))

    state_repeats = sum(1 for state_key, _, _ in test_rows if state_key in train_states)
    identical = sum(
        1 for _, item_key, label in test_rows if train_items.get(item_key) == label
    )
    label_conflicts = sum(
        1
        for _, item_key, label in test_rows
        if item_key in train_items and train_items[item_key] != label
    )
    n_test = len(test_rows)
    return {
        "n_train": n_train,
        "n_test": n_test,
        "state_repeats": state_repeats,
        "identical": identical,
        "label_conflicts": label_conflicts,
        "identical_rate": identical / n_test if n_test else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--max-identical-rate",
        type=float,
        default=None,
        help=(
            "Exit non-zero when a source's identical-item rate exceeds this. "
            "Off by default: the script reports, and gating is the caller's "
            "choice. Use it in a corpus-rebuild check, not as a pass/fail on "
            "data that already exists."
        ),
    )
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    totals = {"n_test": 0, "state_repeats": 0, "identical": 0, "label_conflicts": 0}
    print(f"{'source':<26} {'train':>7} {'test':>7} {'state_dup':>9} {'identical':>9} {'conflict':>8}")
    for path in sorted(args.data_dir.glob("*.jsonl")):
        report = audit_file(path)
        if not report["n_test"]:
            continue
        reports[path.stem] = report
        for key in totals:
            totals[key] += report[key]
        if report["state_repeats"] or report["identical"] or report["label_conflicts"]:
            print(
                f"{path.stem:<26} {report['n_train']:>7} {report['n_test']:>7} "
                f"{report['state_repeats']:>9} {report['identical']:>9} "
                f"{report['label_conflicts']:>8}"
            )

    n_test = totals["n_test"]
    print(
        f"\n{n_test:,} eval items: {totals['state_repeats']:,} repeat a train state "
        f"({100 * totals['state_repeats'] / max(n_test, 1):.2f}%), "
        f"{totals['identical']:,} are the same item with the same label "
        f"({100 * totals['identical'] / max(n_test, 1):.2f}%), "
        f"{totals['label_conflicts']:,} conflict on the label"
    )

    over = []
    if args.max_identical_rate is not None:
        over = [
            (name, report)
            for name, report in reports.items()
            if report["identical_rate"] > args.max_identical_rate
        ]
        for name, report in over:
            print(
                f"FAIL {name}: identical rate {100 * report['identical_rate']:.2f}% "
                f"> {100 * args.max_identical_rate:.2f}%"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_dir": str(args.data_dir),
            "max_identical_rate": args.max_identical_rate,
            "totals": totals,
            "sources": reports,
        }
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Report saved to {args.json}")

    return 0 if not over else 1


if __name__ == "__main__":
    raise SystemExit(main())
