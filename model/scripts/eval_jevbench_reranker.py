#!/usr/bin/env python3
"""Evaluate raw Qwen3-Reranker on JevBench.

Qwen3-Reranker is a causal LM that judges relevance via yes/no logits.
For each jevbench item, we score each option independently by asking
"does this option answer the question for the given situation?" and
taking P(yes). Options are then ranked by their P(yes) scores.

Usage:
    python model/scripts/eval_jevbench_reranker.py --model Qwen/Qwen3-Reranker-4B
    python model/scripts/eval_jevbench_reranker.py --model Qwen/Qwen3-Reranker-0.6B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pipeline import load_jevbench
from model.evaluation.run_provenance import build_run_provenance
from model.src.reranker_prompt import (
    build_query_doc,
    get_options,
    score_options_native,
)

def tier_from_id(item_id: str) -> str:
    return item_id.split("-")[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw Qwen3-Reranker on JevBench")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--variant", choices=["base", "instruct"], default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    snapshot_path = args.snapshot
    if snapshot_path is None:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(args.model))

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(snapshot_path), dtype=torch.bfloat16)
    if device != "auto":
        model = model.to(device)
    model.eval()

    # Prompt template, option extraction and the yes/no read live in
    # model/src/reranker_prompt.py so the general eval harness can score options
    # the same way; this script only decides what to do with the scores.
    max_length = 8192

    print(f"Model loaded on {device}")

    print("Loading JevBench data...")
    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    print(f"  {len(items)} items")

    results_by_tier: dict[str, dict] = {}
    results_by_type: dict[str, dict] = {}
    all_correct = 0
    all_total = 0
    latencies = []
    predictions: list[dict] = []

    for i, item in enumerate(items):
        tier = tier_from_id(item.id)
        q_type = item.question["type"]
        options = get_options(item)

        t0 = time.monotonic()
        scores = score_options_native(
            model, tokenizer, item.state, item.question, options, max_length=max_length, device=device
        )
        elapsed = time.monotonic() - t0
        latencies.append(elapsed)

        prob_dict = {label: round(s, 4) for (label, _), s in zip(options, scores)}

        correct = False
        if q_type == "noul":
            predicted_yes = prob_dict.get("yes", 0) > prob_dict.get("no", 0)
            correct = predicted_yes == item.label
        elif q_type == "choice":
            predicted = max(prob_dict, key=prob_dict.get)
            correct = predicted == item.label
        elif q_type == "score":
            predicted = max(prob_dict, key=prob_dict.get)
            correct = int(predicted) == int(item.label)

        # Record the prediction in the same option order the other JevBench
        # files use: noul true-first, everything else in the criteria order.
        if q_type == "noul":
            order = ["yes", "no"]
            gold_idx = 0 if item.label else 1
            predicted_idx = 0 if prob_dict.get("yes", 0) > prob_dict.get("no", 0) else 1
        elif q_type == "choice":
            order = [label for label, _ in options]
            gold_idx = order.index(item.label) if item.label in order else -1
            predicted_idx = order.index(max(prob_dict, key=prob_dict.get))
        else:
            order = [label for label, _ in options]
            gold_idx = max(0, min(len(order) - 1, int(round(float(item.label)))))
            predicted_idx = int(max(prob_dict, key=prob_dict.get))
        predictions.append(
            {
                "id": item.id,
                "type": q_type,
                "gold_idx": gold_idx,
                "option_count": len(order),
                "p_yes": [round(prob_dict[label], 4) for label in order],
                "predicted_idx": predicted_idx,
                "correct": bool(correct),
            }
        )

        all_total += 1
        if correct:
            all_correct += 1

        tier_data = results_by_tier.setdefault(tier, {"correct": 0, "total": 0})
        tier_data["total"] += 1
        if correct:
            tier_data["correct"] += 1

        type_data = results_by_type.setdefault(q_type, {"correct": 0, "total": 0})
        type_data["total"] += 1
        if correct:
            type_data["correct"] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)} items processed")

    model_name = args.model.split("/")[-1]
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} (raw reranker)")
    print(f"{'=' * 60}")

    print(f"Overall: {all_correct / all_total:.1%} ({all_correct}/{all_total})")

    print("\nBy tier:")
    for tier, data in sorted(results_by_tier.items()):
        acc = data["correct"] / data["total"] if data["total"] else 0
        print(f"  {tier:10s}: {acc:6.1%} ({data['correct']}/{data['total']})")

    print("\nBy type:")
    for qtype, data in sorted(results_by_type.items()):
        acc = data["correct"] / data["total"] if data["total"] else 0
        print(f"  {qtype:10s}: {acc:6.1%} ({data['correct']}/{data['total']})")

    if latencies:
        latencies.sort()
        print(f"\nLatency: mean={sum(latencies)/len(latencies):.2f}s "
              f"median={latencies[len(latencies)//2]:.2f}s total={sum(latencies):.0f}s")

    if args.output:
        provenance = build_run_provenance(
            model_id=args.model,
            snapshot_path=snapshot_path,
            dtype="bfloat16",
            device=str(next(model.parameters()).device),
            strategy="reranker_native",
            norm="none",
            root=ROOT,
            variant=args.variant,
            code_commit=args.code_commit,
            extra={
                "readout": "native P(yes) per option, reranker prompt",
                "scoring_kernel": "model/src/reranker_prompt.py::score_options_native",
                "attn_implementation": getattr(
                    getattr(model, "config", None), "_attn_implementation", None
                ),
                "max_length": max_length,
            },
        )
        summary = {
            "_provenance": provenance,
            "model": args.model,
            "mode": "raw_reranker",
            "items": len(items),
            "n": all_total,
            "failed": len(items) - all_total,
            "overall": {"accuracy": all_correct / all_total, "correct": all_correct, "total": all_total},
            "by_tier": {t: {"accuracy": d["correct"] / d["total"], **d} for t, d in results_by_tier.items()},
            "by_type": {t: {"accuracy": d["correct"] / d["total"], **d} for t, d in results_by_type.items()},
            "predictions": predictions,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
