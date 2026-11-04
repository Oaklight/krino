"""Classify every JevBench schema as seen or held out during training.

A schema is ``(instructions, label set)``, identified by
``model.evaluation.gate.schema_key``.  The training side is the exact corpus a
run config loads: every source in the config, split the way
``data.sources.load_source_items`` splits it.  A JevBench schema that never
appears in that corpus is held out.

Usage:
    python model/scripts/build_schema_split.py \
        --config model/configs/screen_ettin150m.yaml \
        --output data/benchmarks/jevbench/schema_split.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from _vendor import yaml  # zerodep YAML; no PyYAML dependency

from data.pipeline import load_jevbench
from data.sources import load_source_items
from model.evaluation.gate import schema_key


def training_schemas(config_path: Path) -> Counter[str]:
    config = yaml.load(config_path.read_text())
    sources = config.get("sources") or {}
    if not sources:
        raise SystemExit(f"{config_path}: no sources block")
    seen: Counter[str] = Counter()
    for name, source_cfg in sources.items():
        if not isinstance(source_cfg, dict):
            continue
        # Eval-only sources (weight 0 and no training cap) never train, but
        # their schemas still exist in the corpus; count the union.
        rng = random.Random(config.get("seed", 42))
        train_items, eval_items = load_source_items(name, source_cfg, rng)
        for item in list(train_items) + list(eval_items):
            seen[schema_key(item.question)] += 1
        print(f"  {name:16s} train={len(train_items):6d} eval={len(eval_items):6d}", flush=True)
    return seen


def jevbench_schemas() -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in load_jevbench():
        counts[schema_key(item.question)] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Training corpus from {args.config}", flush=True)
    train = training_schemas(args.config)
    bench = jevbench_schemas()

    split = {key: ("seen" if key in train else "held_out") for key in bench}
    by_label = Counter(split.values())
    items_by_label = Counter()
    for key, label in split.items():
        items_by_label[label] += bench[key]

    payload = {
        "config": str(args.config),
        "n_train_schemas": len(train),
        "n_jevbench_schemas": len(bench),
        "schemas": dict(sorted(split.items())),
        "summary": {
            "schemas": dict(by_label),
            "items": dict(items_by_label),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(
        f"JevBench: {len(bench)} schemas, {sum(bench.values())} items -> "
        f"{dict(by_label)}; wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
