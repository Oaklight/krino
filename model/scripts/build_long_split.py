"""Select the long-context items the residual arms cannot see at 2048 tokens.

The 5,000-item cache has p99 = 2572 tokens, so a `max_length=2048` arm never
trains on the long hard items that carry the JevBench hard tier. This script
loads the same corpus a training config loads, filters by a character budget
first (cheap), then confirms the real token count with the arm's tokenizer.

Usage:
    python model/scripts/build_long_split.py \
        --config model/configs/screen_ettin150m.yaml \
        --out data/benchmarks/residual_long.jsonl --limit 3000
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from _vendor import yaml  # zerodep YAML; no PyYAML dependency
from transformers import AutoTokenizer

from data.pipeline import save_jsonl
from data.sources import load_source_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--char-prefilter", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.load(args.config.read_text())
    seed = int(config.get("seed", 42))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    candidates = []
    seen: set[str] = set()
    for name, cfg in (config.get("sources") or {}).items():
        if not isinstance(cfg, dict):
            continue
        rng = random.Random(seed)
        train_items, eval_items = load_source_items(name, cfg, rng)
        for item in list(train_items) + list(eval_items):
            if item.id in seen:
                continue
            text = f"{item.state} {item.question.get('instructions', '')}"
            if len(text) < args.char_prefilter:
                continue
            seen.add(item.id)
            candidates.append((name, item, text))
    print(f"char-prefilter candidates: {len(candidates)}", flush=True)

    kept = []
    by_source = collections.Counter()
    for name, item, text in candidates:
        n_tokens = len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        if n_tokens > args.min_tokens:
            kept.append((name, item, n_tokens))
            by_source[name] += 1
    print(f"over {args.min_tokens} tokens: {len(kept)}", flush=True)
    print("by source:", dict(by_source.most_common(10)), flush=True)

    random.Random(args.seed).shuffle(kept)
    kept = kept[: args.limit]
    types = collections.Counter(item.question["type"] for _, item, _ in kept)
    lengths = sorted(n for _, _, n in kept)
    print(
        f"kept {len(kept)} types={dict(types)} "
        f"tokens p50={lengths[len(lengths)//2]} max={lengths[-1]}",
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl([item for _, item, _ in kept], args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
