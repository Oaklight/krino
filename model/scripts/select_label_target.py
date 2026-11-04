"""Select a fixed-size label target from band-selected items.

``band_selection.py`` returns every item whose p(gold) lies in the band.
Labeling all of them may exceed the agreed budget and may not match the type
mix the arm needs.  This script excludes ids that are already labeled or
already in a training split, then draws a fixed total stratified by question
type.  Selection is deterministic for a given seed.

Usage:
    python model/scripts/select_label_target.py \
        --band data/benchmarks/band_full_u50.jsonl \
        --exclude data/benchmarks/residual_train_20000.jsonl \
        --out data/benchmarks/label_target_30k.jsonl \
        --total 30000 --shares noul=0.32,choice=0.60,score=0.08 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.pipeline import load_jsonl, save_jsonl

TYPES = ("noul", "choice", "score")


def load_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for item in load_jsonl(path):
            ids.add(item.id)
    return ids


def parse_shares(text: str) -> dict[str, float]:
    shares: dict[str, float] = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in TYPES:
            raise SystemExit(f"unknown type in --shares: {key}")
        shares[key] = float(value)
    if not shares:
        raise SystemExit("--shares is empty")
    total = sum(shares.values())
    if total <= 0:
        raise SystemExit("--shares must sum to a positive value")
    return {key: value / total for key, value in shares.items()}


def allocate(total: int, shares: dict[str, float], available: dict[str, int]) -> dict[str, int]:
    """Largest-remainder quotas, then fill any shortfall from spare capacity."""
    raw = {key: total * share for key, share in shares.items()}
    quota = {key: min(int(raw[key]), available.get(key, 0)) for key in shares}
    remainder = total - sum(quota.values())
    for key in sorted(shares, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if remainder <= 0:
            break
        if quota[key] < available.get(key, 0):
            quota[key] += 1
            remainder -= 1

    # Shortfall from a type with too few items goes to types with capacity.
    while remainder > 0:
        spare = {
            key: available.get(key, 0) - quota[key]
            for key in shares
            if available.get(key, 0) > quota[key]
        }
        if not spare:
            break
        for key in sorted(spare):
            if remainder <= 0:
                break
            if spare[key] > 0:
                quota[key] += 1
                remainder -= 1
    return quota


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="JSONL files whose ids must not be selected",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument(
        "--shares",
        default="noul=0.32,choice=0.60,score=0.08",
        help="Target type mix, normalised before allocation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    shares = parse_shares(args.shares)
    band = load_jsonl(args.band)
    if args.limit is not None:
        band = band[: args.limit]
    excluded = load_ids(args.exclude)
    eligible = [item for item in band if item.id not in excluded]
    skipped_augmented = sum(
        1
        for item in eligible
        if item.question["type"] == "noul" and item.question.get("augmented_options")
    )

    groups: dict[str, list] = defaultdict(list)
    for item in eligible:
        q_type = item.question["type"]
        if q_type not in shares:
            continue
        if q_type == "noul" and item.question.get("augmented_options"):
            continue
        groups[q_type].append(item)

    available = {key: len(groups[key]) for key in shares}
    quota = allocate(args.total, shares, available)

    rng = random.Random(args.seed)
    selected = []
    for key in shares:
        group = list(groups[key])
        rng.shuffle(group)
        selected.extend(group[: quota[key]])
    rng.shuffle(selected)

    save_jsonl(selected, args.out)
    stats = {
        "band": str(args.band),
        "exclude": [str(path) for path in args.exclude],
        "out": str(args.out),
        "total_requested": args.total,
        "band_items": len(band),
        "excluded_ids": len(excluded),
        "eligible": len(eligible),
        "skipped_augmented": skipped_augmented,
        "shares": shares,
        "available_by_type": available,
        "quota_by_type": quota,
        "selected_by_type": dict(Counter(item.question["type"] for item in selected)),
        "selected": len(selected),
        "shortfall": args.total - len(selected),
        "selected_by_source": dict(
            Counter(item.source for item in selected).most_common(20)
        ),
    }
    payload = json.dumps(stats, indent=1, sort_keys=True)
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(payload + "\n")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
