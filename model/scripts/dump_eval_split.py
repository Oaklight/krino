"""Materialise the eval split a training config uses, as one JSONL file.

Used to score a reference model on the same items the trained arms are
evaluated on, so the benchmark-vs-corpus gap can be measured on one item set.

Usage:
    python model/scripts/dump_eval_split.py --config model/configs/screen_ettin150m.yaml \
        --max-items 1000 --output /tmp/eval_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from data.sources import load_source_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated source names to keep; empty keeps all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    rng = random.Random(args.seed)
    keep = {name.strip() for name in args.sources.split(",") if name.strip()}
    items = []
    sources = config.get("sources") or {}
    for name, source_cfg in sources.items():
        if not isinstance(source_cfg, dict):
            continue
        if keep and name not in keep:
            continue
        train_items, eval_items = load_source_items(name, source_cfg, rng)
        items.extend(train_items if args.split == "train" else eval_items)
    rng.shuffle(items)
    items = items[: args.max_items]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} {args.split} items to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
