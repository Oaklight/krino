"""Cache the base reranker's option distribution for a set of training items.

The 4B residual arm needs ``p_base`` per item before training, because one 4B
forward per (context, option) pair dominates the cost.  This writes one JSON
record per item: ``{"id", "q_type", "keys", "probs"}``.

Runs are resumable: ids already present in the output file are skipped.

Usage:
    python model/scripts/precompute_base_probs.py \
        --items data/benchmarks/residual_train_5000.jsonl \
        --model Qwen/Qwen3-Reranker-4B --output data/benchmarks/base_probs_5000.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    done: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"resuming: {len(done)} items already cached", flush=True)

    items = [item for item in load_jsonl(args.items) if item.id not in done]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"{len(items)} items to score", flush=True)

    scorer = RawReranker(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(items, 1):
            q = item.question
            options = _raw_options(q)
            if len(options) < 2:
                continue
            scores = [
                scorer._score_option(item.state, q.get("instructions", ""), q["type"], label, desc)
                for label, desc in options
            ]
            total = sum(scores) or 1.0
            record = {
                "id": item.id,
                "q_type": q["type"],
                "keys": [label for label, _ in options],
                "probs": [score / total for score in scores],
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1
            if index % 50 == 0:
                elapsed = time.time() - started
                rate = index / elapsed
                eta = (len(items) - index) / rate if rate else 0
                print(
                    f"  {index}/{len(items)}  {rate:.1f} items/s  eta {eta/60:.0f} min",
                    flush=True,
                )
    print(f"wrote {written} records to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
