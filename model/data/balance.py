"""Balance the training data distribution across question types and sources."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from .pipeline import DATA_DIR, load_jsonl, save_jsonl

DEFAULT_CAPS: dict[str, int] = {
    "mnli": 30_000,
    "sst2": 30_000,
    "tabfact": 40_000,
    "fever": 30_000,
    "race": 30_000,
    "hellaswag": 30_000,
    "agnews": 30_000,
}


def count_distribution(items: list) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        q_type = item.question["type"]
        dist[q_type][item.source] += 1
    return {k: dict(v) for k, v in dist.items()}


def downsample(
    items: list, caps: dict[str, int], seed: int = 42
) -> list:
    rng = random.Random(seed)
    by_source: dict[str, list] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    result = []
    for source, source_items in by_source.items():
        cap = caps.get(source)
        if cap is not None and len(source_items) > cap:
            result.extend(rng.sample(source_items, cap))
        else:
            result.extend(source_items)
    return result


def print_report(before: dict, after: dict) -> None:
    print("\n  Distribution report:")
    print(f"  {'Type':<10} {'Source':<20} {'Before':>10} {'After':>10}")
    print(f"  {'-'*10} {'-'*20} {'-'*10} {'-'*10}")
    for q_type in sorted(set(list(before.keys()) + list(after.keys()))):
        sources = sorted(set(
            list(before.get(q_type, {}).keys()) +
            list(after.get(q_type, {}).keys())
        ))
        for source in sources:
            b = before.get(q_type, {}).get(source, 0)
            a = after.get(q_type, {}).get(source, 0)
            print(f"  {q_type:<10} {source:<20} {b:>10,} {a:>10,}")
        type_before = sum(before.get(q_type, {}).values())
        type_after = sum(after.get(q_type, {}).values())
        print(f"  {q_type:<10} {'TOTAL':<20} {type_before:>10,} {type_after:>10,}")
        print()

    total_before = sum(sum(v.values()) for v in before.values())
    total_after = sum(sum(v.values()) for v in after.values())
    print(f"  {'ALL':<10} {'TOTAL':<20} {total_before:>10,} {total_after:>10,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Balance training data distribution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't save")
    args = parser.parse_args()

    jsonl_files = sorted(DATA_DIR.rglob("*.jsonl"))
    # Exclude balanced outputs and raw/unconverted files
    jsonl_files = [
        f for f in jsonl_files
        if not f.stem.startswith("balanced_") and "_raw" not in f.stem
    ]

    all_items = []
    for path in jsonl_files:
        items = load_jsonl(path)
        print(f"  loaded {len(items):>8,} items from {path.relative_to(DATA_DIR)}")
        all_items.extend(items)

    before = count_distribution(all_items)
    balanced = downsample(all_items, DEFAULT_CAPS, seed=args.seed)
    after = count_distribution(balanced)

    print_report(before, after)

    if args.dry_run:
        print("\n  (dry run — no files saved)")
        return 0

    train_items = [item for item in balanced if item.split == "train"]
    test_items = [item for item in balanced if item.split == "test"]

    train_path = DATA_DIR / "balanced_train.jsonl"
    test_path = DATA_DIR / "balanced_test.jsonl"

    n_train = save_jsonl(train_items, train_path)
    n_test = save_jsonl(test_items, test_path)
    print(f"\n  → saved {n_train:,} train items to {train_path}")
    print(f"  → saved {n_test:,} test items to {test_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
