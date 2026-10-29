#!/usr/bin/env python3
"""Compare two checkpoints item by item on the JevBench hard tier.

The public hard tier is 111 items, 67 of them choice, so a single checkpoint's
accuracy there is a noisy estimate of the model: the matched pair below puts the
same readout 13 points apart on seeds alone. This asks the questions that need
two checkpoints to answer -- how many items do both get, how many does either
get, and is the overlap better than two independent answerers would produce --
and writes the per-item outcomes so the counts can be checked against the tree.

Both readouts are read straight off a ``blend_scorers.py --dump-records`` file,
so this needs no GPU. The blend itself is not scored here; it lives in the PR
that introduced it.

The dump records carry no item id, only the group ``jevbench-hard`` and the
order they were produced in, which is the order of the choice items in
``hard.jsonl``. That assumption is checked, not trusted: the record count must
equal the number of choice items and every record's option count must match its
item's.

Usage:
    python model/scripts/jevbench_cross_seed.py \
        --records model/experiments/blend_records_full_s123_ep5.json \
        --records model/experiments/blend_records_full_s42_ep5.json \
        --json model/experiments/jevbench_cross_seed_hard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TIER = "jevbench-hard"


def argmax(scores: Sequence[float]) -> int:
    """Index of the highest score, first one wins ties."""
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def load_tier(records_path: Path) -> list[dict[str, Any]]:
    """The hard-tier records of one checkpoint, in file order."""
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    return [record for record in payload["records"] if record["group"] == TIER]


def load_choice_items(data_root: Path) -> list[dict[str, Any]]:
    """Choice items of the public hard tier, in file order."""
    path = data_root / "data" / "benchmarks" / "jevbench" / "hard.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item for item in items if item["question"].get("type") == "choice"]


def align(records: list[dict[str, Any]], items: list[dict[str, Any]], label: str) -> None:
    """Fail loudly if the records are not one per choice item, in order."""
    if len(records) != len(items):
        raise SystemExit(
            f"Error: {label} has {len(records)} hard-tier records against {len(items)} choice items; "
            "the positional mapping is only valid when every item was scored"
        )
    for index, (record, item) in enumerate(zip(records, items)):
        options = item["question"].get("criteria") or item["labels"]
        if record["k"] != len(options):
            raise SystemExit(
                f"Error: {label} record {index} ({item['id']}) has {record['k']} options "
                f"against {len(options)} in the item; the two files are not in the same order"
            )


def outcomes(records: list[dict[str, Any]], readout: str) -> list[bool]:
    """Per-item correctness of one readout."""
    return [
        argmax(record[readout]) == record["gold"] for record in records
    ]


def summarise(first: list[bool], second: list[bool]) -> dict[str, Any]:
    """Counts and rates for one pair of checkpoints."""
    n = len(first)
    both = sum(1 for left, right in zip(first, second) if left and right)
    either = sum(1 for left, right in zip(first, second) if left or right)
    rate_first = sum(first) / n
    rate_second = sum(second) / n
    return {
        "n": n,
        "correct": [sum(first), sum(second)],
        "accuracy": [round(rate_first, 4), round(rate_second, 4)],
        "both_right": both,
        "either_right": either,
        "either_accuracy": round(either / n, 4),
        "disagree": sum(1 for left, right in zip(first, second) if left != right),
        # What the overlap would be if the two checkpoints answered independently
        # at their own rates -- the null the both-right count is read against.
        "independent_overlap": round(n * rate_first * rate_second, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        action="append",
        required=True,
        help="blend_scorers.py --dump-records file; pass one per checkpoint, in the order to report",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if len(args.records) != 2:
        raise SystemExit("Error: exactly two --records files are needed for a cross-checkpoint comparison")

    items = load_choice_items(args.data_root)
    tiers = []
    for index, path in enumerate(args.records):
        records = load_tier(path)
        align(records, items, f"checkpoint {index} ({path.name})")
        tiers.append(records)

    results: dict[str, Any] = {}
    per_item: dict[str, Any] = {}
    for readout in ("native", "head"):
        first = outcomes(tiers[0], readout)
        second = outcomes(tiers[1], readout)
        results[readout] = summarise(first, second)
        per_item[readout] = [bool(value) for value in first], [bool(value) for value in second]
        row = results[readout]
        print(
            f"{readout:<8} {row['correct'][0]}/{row['n']} vs {row['correct'][1]}/{row['n']}  "
            f"both {row['both_right']} (indep {row['independent_overlap']})  "
            f"either {row['either_right']} ({row['either_accuracy']:.4f})  disagree {row['disagree']}",
            flush=True,
        )

    if args.json:
        payload = {
            "tier": TIER,
            "checkpoints": [str(path) for path in args.records],
            "items": [item["id"] for item in items],
            "families": [item["family"] for item in items],
            "results": results,
            "per_item": {
                readout: {"first": first, "second": second}
                for readout, (first, second) in per_item.items()
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
