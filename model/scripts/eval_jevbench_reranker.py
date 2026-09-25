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

INSTRUCTION = (
    "Classify: which option correctly answers the question for the given situation? "
    "Rate how well each option fits."
)


def build_query_doc(item, label: str, description: str) -> str:
    """Build a query-document pair in Qwen3-Reranker's expected format."""
    q = item.question
    query_text = f"{item.state}\n\nQuestion: {q['instructions']}"
    q_type = q["type"]
    if q_type == "score":
        doc_text = f"Rubric level {label}: {description}"
    else:
        doc_text = f"Answer option {label}: {description}"

    return (
        f"<Instruct>: {INSTRUCTION}\n"
        f"<Query>: {query_text}\n"
        f"<Document>: {doc_text}"
    )


def get_options(item) -> list[tuple[str, str]]:
    """Return (label, description) pairs for each answer option."""
    q = item.question
    criteria = q.get("criteria", {})
    if q["type"] == "noul":
        return [
            ("no", str(criteria.get("false", criteria.get("no", "No")))),
            ("yes", str(criteria.get("true", criteria.get("yes", "Yes")))),
        ]
    elif q["type"] == "choice":
        return [(k, str(v)) for k, v in criteria.items()]
    elif q["type"] == "score":
        if isinstance(criteria, list):
            return [(str(i), str(desc)) for i, desc in enumerate(criteria)]
        return [(k, str(v)) for k, v in criteria.items()]
    return []


def tier_from_id(item_id: str) -> str:
    return item_id.split("-")[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw Qwen3-Reranker on JevBench")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if device != "auto":
        model = model.to(device)
    model.eval()

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
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

    for i, item in enumerate(items):
        tier = tier_from_id(item.id)
        q_type = item.question["type"]
        options = get_options(item)

        t0 = time.monotonic()
        scores = []
        for label, desc in options:
            text = build_query_doc(item, label, desc)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens = tokens[: max_length - len(prefix_tokens) - len(suffix_tokens)]
            input_ids = prefix_tokens + tokens + suffix_tokens
            inputs = {
                "input_ids": torch.tensor([input_ids], device=device),
                "attention_mask": torch.ones(1, len(input_ids), device=device),
            }
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
            true_logit = logits[:, token_true_id].item()
            false_logit = logits[:, token_false_id].item()
            score = torch.softmax(
                torch.tensor([false_logit, true_logit]), dim=0
            )[1].item()
            scores.append(score)

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
        summary = {
            "model": args.model,
            "mode": "raw_reranker",
            "overall": {"accuracy": all_correct / all_total, "correct": all_correct, "total": all_total},
            "by_tier": {t: {"accuracy": d["correct"] / d["total"], **d} for t, d in results_by_tier.items()},
            "by_type": {t: {"accuracy": d["correct"] / d["total"], **d} for t, d in results_by_type.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
