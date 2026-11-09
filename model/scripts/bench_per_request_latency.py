#!/usr/bin/env python3
"""Latency of the per-request judgement layout, with and without prefix reuse (#324).

Single-item latency is one forward either way, so it cannot show the cost Max
names.  The cost appears on the second question over the same context: a causal
pass can reuse the state's KV cache, a full-attention pass cannot, because the
prefix tokens would have to attend to the new suffix.

Reported per item: the full-sequence forward (the single-item latency), the
one-off state-prefix forward, and the suffix forward against the cached prefix.
For ``--direction full`` the cached numbers are ``null``: reuse is invalid.

Usage:
    python model/scripts/bench_per_request_latency.py \\
        --model Qwen/Qwen3-Reranker-4B --direction causal --output lat.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.scripts.eval_jevbench_judgement_per_request import build_sequence  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402


def time_forward(tower, *args, **kwargs) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        tower(*args, **kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--direction", choices=["causal", "full"], required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(args.model))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_causal_lm(str(snapshot), device=None, dtype=dtype)
    tower = getattr(model, "model", model)
    device = str(next(model.parameters()).device)

    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]

    # Warm-up so the first measured item is not paying for kernel selection.
    warm = tokenizer(
        "State: warm up.\nQuestion: warm up.\nOption: a\nDoes option 1 correctly answer the question? Answer:",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    time_forward(tower, input_ids=warm, use_cache=False)

    rows = []
    for item in items:
        built = build_sequence(item)
        if built is None:
            continue
        text, char_ends, _ = built
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)

        full = time_forward(
            tower, input_ids=input_ids, attention_mask=attention, use_cache=False
        )

        prefix_s = None
        suffix_s = None
        if args.direction == "causal":
            split = text.index("\n")
            prefix_ids = tokenizer(
                text[:split], add_special_tokens=False, return_tensors="pt"
            )
            suffix_ids = tokenizer(
                text[split:], add_special_tokens=False, return_tensors="pt"
            )
            prefix = prefix_ids["input_ids"].to(device)
            suffix = suffix_ids["input_ids"].to(device)
            mask = torch.ones(
                (1, prefix.shape[1] + suffix.shape[1]), dtype=torch.long, device=device
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                cached = tower(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    use_cache=True,
                )
            torch.cuda.synchronize()
            prefix_s = time.perf_counter() - started
            suffix_s = time_forward(
                tower,
                input_ids=suffix,
                attention_mask=mask,
                past_key_values=cached.past_key_values,
                use_cache=True,
            )
        rows.append(
            {
                "id": item.id,
                "full_s": full,
                "prefix_s": prefix_s,
                "suffix_s": suffix_s,
            }
        )

    def summary(key):
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            return None
        return {
            "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "n": len(values),
        }

    result = {
        "direction": args.direction,
        "model": args.model,
        "items": len(rows),
        "single_item": summary("full_s"),
        "state_prefix": summary("prefix_s"),
        "second_question_cached_suffix": summary("suffix_s"),
        "note": (
            "second_question_cached_suffix is the cost of a second question over "
            "the same state under a causal mask; under a full mask the cache is "
            "not reusable and the second question costs single_item again"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**result, "rows": rows}, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
