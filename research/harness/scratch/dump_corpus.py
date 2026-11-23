"""Materialise every item the training config would load, for band selection."""
import argparse
import collections
import random
from pathlib import Path

import yaml

from data.pipeline import save_jsonl
from data.sources import load_source_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    seed = int(config.get("seed", 42))
    items = []
    seen = set()
    for name, cfg in (config.get("sources") or {}).items():
        if not isinstance(cfg, dict):
            continue
        rng = random.Random(seed)
        train_items, eval_items = load_source_items(name, cfg, rng)
        for item in list(train_items) + list(eval_items):
            if item.id in seen:
                continue
            if item.question["type"] not in ("noul", "choice", "score"):
                continue
            seen.add(item.id)
            items.append(item)
    save_jsonl(items, args.out)
    kinds = collections.Counter(i.question["type"] for i in items)
    print(f"wrote {len(items)} items to {args.out} {dict(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
