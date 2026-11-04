"""Grow the residual training split without resampling what is already cached.

The B1/B2 arms trained on 5,000 items.  research/11's design calls for a
20-40K subset, so this script keeps the existing 5,000 exactly as they are and
draws the remainder from the same corpus, stratified by (type, source) so the
type mix matches the corpus instead of drifting.

Usage:
    python model/scripts/build_residual_split.py \
        --config model/configs/screen_ettin150m.yaml \
        --base data/benchmarks/residual_train_5000.jsonl \
        --out data/benchmarks/residual_train_20000.jsonl --total 20000
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

from _vendor import yaml  # zerodep YAML; no PyYAML dependency

from data.pipeline import load_jsonl, save_jsonl
from data.sources import load_source_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_items = load_jsonl(args.base)
    have = {item.id for item in base_items}
    need = args.total - len(base_items)
    if need <= 0:
        raise SystemExit("base already covers --total")

    config = yaml.load(args.config.read_text())
    seed = int(config.get("seed", 42))

    pools: dict[tuple[str, str], list] = collections.defaultdict(list)
    for name, cfg in (config.get("sources") or {}).items():
        if not isinstance(cfg, dict):
            continue
        rng = random.Random(seed)
        train_items, eval_items = load_source_items(name, cfg, rng)
        for item in list(train_items) + list(eval_items):
            if item.id in have:
                continue
            if item.question["type"] not in ("noul", "choice", "score"):
                continue
            pools[(name, item.question["type"])].append(item)

    total_available = sum(len(v) for v in pools.values())
    print(f"base {len(base_items)}, need {need}, pool {total_available}", flush=True)

    rng = random.Random(args.seed)
    picked = []
    keys = sorted(pools)
    # Proportional draw by pool size, then top up from whatever is left.
    for key in keys:
        share = int(round(need * len(pools[key]) / total_available))
        group = pools[key]
        rng.shuffle(group)
        picked.extend(group[:share])
    if len(picked) < need:
        chosen = {item.id for item in picked}
        leftovers = [
            item
            for key in keys
            for item in pools[key]
            if item.id not in chosen
        ]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: need - len(picked)])
    picked = picked[:need]

    combined = list(base_items) + picked
    types = collections.Counter(item.question["type"] for item in combined)
    print(f"total {len(combined)} types={dict(types)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(combined, args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
