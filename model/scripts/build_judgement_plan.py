#!/usr/bin/env python3
"""Build the judgement-objective prompt plan from the canonical corpus (#324).

One JSONL line per option prompt, produced by
``model/src/judgement_objective.py::build_prompts``, which itself calls the
same ``candidates``/``render`` the readout scores.  Training and inference
therefore share one definition (``#299``).

The canonical corpus is the source list of ``model/configs/r2_reranker4b.yaml``
with the per-source ``max_train`` caps removed unless ``--capped`` is passed:
that is the 243,366-row corpus ``#299`` measured.  Abstention rows
(``answerable = False``) are excluded; eval-only sources (``weight: 0``) are
kept out of the train split.

Usage:
    python model/scripts/build_judgement_plan.py \\
        --config model/configs/r2_reranker4b.yaml \\
        --out data/benchmarks/judgement_plan.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _vendor import yaml  # noqa: E402

from data.sources import load_source_items  # noqa: E402
from model.src.judgement_objective import build_prompts  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "model/configs/r2_reranker4b.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capped",
        action="store_true",
        help="keep each source's max_train/max_eval caps instead of the full split",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="debug aid: keep at most this many rows per source",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    sources: dict[str, dict] = config.get("sources", {})
    if not sources:
        raise SystemExit(f"{args.config} has no sources")

    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    rows = 0
    prompts = 0
    skipped_items = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for name, raw_cfg in sources.items():
            cfg = dict(raw_cfg)
            if not args.capped:
                cfg.pop("max_train", None)
                cfg.pop("max_eval", None)
            train_items, eval_items = load_source_items(name, cfg, rng)
            items = train_items + eval_items
            if args.max_rows_per_source is not None:
                items = items[: args.max_rows_per_source]
            kept = 0
            for item in items:
                if dataclasses.is_dataclass(item) and getattr(item, "answerable", True) is False:
                    continue
                split = getattr(item, "split", "train")
                row_id = f"{name}:{item.id}"
                built = build_prompts(item, row_id=row_id, source=name, split=split)
                if not built:
                    skipped_items += 1
                    continue
                kept += 1
                rows += 1
                counts[f"{split}:{item.question.get('type', 'noul')}"] += 1
                if len(built) == 1 and not built[0].has_option_span:
                    counts[f"{split}:polarity"] += 1
                for prompt in built:
                    record = dataclasses.asdict(prompt)
                    record["label"] = prompt.label
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    prompts += 1
            print(f"{name:22} rows {kept:7,}", flush=True)

    print(f"wrote {args.out}")
    print(f"rows {rows:,}  prompts {prompts:,}  items skipped {skipped_items:,}")
    for key in sorted(counts):
        print(f"  {key:20} {counts[key]:8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
