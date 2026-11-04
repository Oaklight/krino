"""Measure how many items the arm truncates at each candidate max_length.

``DecisionModel._tokenize_context`` tokenizes ``state + instructions`` and,
when the sequence is too long, keeps the instructions and truncates the state
from the left.  The arm then trains on that shortened prefix even though the
label was computed from the full context.  This script reproduces the decision
without loading a model, so a length arm can be sized before it is registered.

Usage:
    python model/scripts/truncation_audit.py \
        --items data/benchmarks/residual_train_20000.jsonl \
        --tokenizer Qwen/Qwen3-Reranker-4B \
        --max-length 2048 4096 \
        --stats data/benchmarks/truncation_20000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from data.pipeline import load_jsonl, save_jsonl


def _count(tokenizer, text: str, add_special_tokens: bool = True) -> int:
    encoded = tokenizer(
        text, add_special_tokens=add_special_tokens, truncation=False
    )
    return len(encoded["input_ids"])


def _dropped_state(tokenizer, state: str, instructions: str, max_length: int) -> tuple[int, bool]:
    """State tokens removed by the fallback, and whether instructions overflow."""
    instruction_ids = tokenizer(" " + instructions, add_special_tokens=False)["input_ids"]
    state_ids = tokenizer(state, add_special_tokens=False)["input_ids"]
    if len(instruction_ids) >= max_length:
        return len(state_ids), True
    state_budget = max_length - len(instruction_ids)
    dropped = max(0, len(state_ids) - state_budget)
    return dropped, False


def percentiles(values: list[int], qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for q in qs:
        position = min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))
        out[f"p{q}"] = ordered[position]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument(
        "--max-length",
        type=int,
        nargs="+",
        default=[2048],
        help="One or more context budgets to audit",
    )
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument(
        "--emit-length",
        type=int,
        default=None,
        help="Length used by --out-truncated / --out-kept (default: smallest)",
    )
    parser.add_argument("--out-truncated", type=Path, default=None)
    parser.add_argument("--out-kept", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    lengths = sorted(set(args.max_length))
    emit_length = args.emit_length or lengths[0]
    if emit_length not in lengths:
        raise SystemExit("--emit-length must be one of --max-length")

    items = load_jsonl(args.items)
    if args.limit is not None:
        items = items[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    combined_lengths: list[int] = []
    per_length = {
        length: {
            "counts": Counter(),
            "by_type": defaultdict(Counter),
            "by_source": defaultdict(Counter),
            "dropped_state": [],
        }
        for length in lengths
    }
    truncated_items = []
    kept_items = []

    for index, item in enumerate(items, 1):
        state = str(item.state)
        instructions = item.question.get("instructions", "")
        combined = _count(tokenizer, f"{state} {instructions}", add_special_tokens=True)
        combined_lengths.append(combined)

        q_type = item.question["type"]
        emit_truncated = False
        for length in lengths:
            bucket = per_length[length]
            bucket["counts"]["input"] += 1
            bucket["by_type"][q_type]["input"] += 1
            bucket["by_source"][item.source]["input"] += 1
            if combined <= length:
                continue
            dropped, instruction_overflow = _dropped_state(
                tokenizer, state, instructions, length
            )
            bucket["counts"]["truncated"] += 1
            bucket["counts"]["instruction_overflow"] += int(instruction_overflow)
            bucket["by_type"][q_type]["truncated"] += 1
            bucket["by_source"][item.source]["truncated"] += 1
            bucket["dropped_state"].append(dropped)
            if length == emit_length:
                emit_truncated = True

        if emit_truncated:
            truncated_items.append(item)
        else:
            kept_items.append(item)

        if index % 5000 == 0:
            rate = per_length[emit_length]["counts"]["truncated"] / index
            print(
                f"  {index}/{len(items)} truncated@{emit_length}={rate:.3f}",
                flush=True,
            )

    stats = {
        "items": str(args.items),
        "tokenizer": args.tokenizer,
        "max_lengths": lengths,
        "n": len(items),
        "combined_tokens": percentiles(combined_lengths),
        "max_combined_tokens": max(combined_lengths) if combined_lengths else 0,
        "per_length": {},
    }
    for length in lengths:
        bucket = per_length[length]
        counts = bucket["counts"]
        stats["per_length"][str(length)] = {
            "counts": dict(counts),
            "rate": counts["truncated"] / max(1, counts["input"]),
            "by_type": {
                q_type: dict(counter)
                for q_type, counter in sorted(bucket["by_type"].items())
            },
            "by_source": {
                source: dict(counter)
                for source, counter in sorted(bucket["by_source"].items())
            },
            "dropped_state_tokens": percentiles(bucket["dropped_state"]),
        }

    payload = json.dumps(stats, indent=1, sort_keys=True)
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(payload + "\n")
    print(payload, flush=True)

    if args.out_truncated:
        save_jsonl(truncated_items, args.out_truncated)
        print(f"wrote {len(truncated_items)} truncated items to {args.out_truncated}", flush=True)
    if args.out_kept:
        save_jsonl(kept_items, args.out_kept)
        print(f"wrote {len(kept_items)} kept items to {args.out_kept}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
