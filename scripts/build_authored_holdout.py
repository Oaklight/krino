#!/usr/bin/env python3
"""Write the deterministic authored held-out set to JSONL.

The set exists because clause 4's public tier is too small to decide anything below
15.7 points (`research/23`), and the benchmark does not publish more items. Every item
is generated from a numeric specification and its gold label is computed from that
specification, so no label depends on a judgement call.

Usage:
    python scripts/build_authored_holdout.py
    python scripts/build_authored_holdout.py --per-family 51 --out data/benchmarks/openjev/authored_holdout.jsonl

The output is data, not code, and lives beside `authored144.jsonl` where the other
authored sets live. It is not committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.authored_holdout import build_items, check_item  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "benchmarks" / "openjev" / "authored_holdout.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-family", type=int, default=51)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    items = list(build_items(per_family=args.per_family, seed=args.seed))
    problems = [(item["id"], issue) for item in items for issue in check_item(item)]
    if problems:
        for item_id, issue in problems:
            print(f"FAIL {item_id}: {issue}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    by_type = Counter(item["question"]["type"] for item in items)
    by_group = Counter(item["group"] for item in items)
    print(f"wrote {len(items)} items to {args.out}", flush=True)
    print(f"  by type:  {dict(by_type)}", flush=True)
    print(f"  by group: {dict(by_group)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
