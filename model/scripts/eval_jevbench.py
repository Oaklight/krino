#!/usr/bin/env python3
"""Evaluate Krino HuggingFace models on JevBench.

Downloads jevbench public data (easy/hard/original tiers), loads a Krino
model from HuggingFace, runs inference, and reports accuracy per tier
and question type.

Usage:
    python model/scripts/eval_jevbench.py --repo oaklight/krino-qwen3-0.6b-heads
    python model/scripts/eval_jevbench.py --repo oaklight/krino-ettin-150m-heads --device cpu
    python model/scripts/eval_jevbench.py --all --output results/jevbench/
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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from data.pipeline import load_jevbench
from model.evaluation.accuracy import choice_accuracy, noul_accuracy
from model.src.backbone import load_causal_lm, load_encoder, load_qwen35_base
from model.src.decision_model import DecisionModel

KRINO_REPOS = [
    "oaklight/krino-modernbert-base-heads",
    "oaklight/krino-ettin-150m-heads",
    "oaklight/krino-qwen3-0.6b-heads",
    "oaklight/krino-qwen3-reranker-0.6b-heads",
    "oaklight/krino-qwen3-reranker-4b-heads",
    "oaklight/krino-qwen3.5-4b-heads",
]


def load_krino_from_hf(
    repo_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> DecisionModel:
    """Load a Krino model from HuggingFace."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    backbone_name = config["backbone"]["name"]
    loader_type = config["backbone"].get("loader", "causal_lm")
    rank = config["heads"]["rank"]
    dropout = config["heads"]["dropout"]

    print(f"  Loading backbone: {backbone_name} ({loader_type})")
    if loader_type == "encoder":
        backbone, tokenizer = load_encoder(backbone_name, device=device, dtype=dtype)
    elif loader_type == "qwen35_base":
        backbone, tokenizer = load_qwen35_base(backbone_name, device=device, dtype=dtype)
    else:
        backbone, tokenizer = load_causal_lm(backbone_name, device=device, dtype=dtype)

    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=rank,
        dropout=dropout,
    )

    weights_path = hf_hub_download(repo_id, "head_weights.safetensors")
    head_state = load_file(weights_path, device=device)
    model.load_state_dict(head_state, strict=False)
    model.eval()
    print(f"  Heads loaded ({sum(v.numel() for v in head_state.values()):,} params)")
    return model


def tier_from_id(item_id: str) -> str:
    return item_id.split("-")[0]


def evaluate_jevbench(model: DecisionModel, items: list) -> dict:
    """Run jevbench eval and return per-tier and overall results."""
    results_by_tier: dict[str, dict] = {}
    results_by_type: dict[str, dict] = {}
    all_correct = 0
    all_total = 0
    all_failed = 0
    latencies = []

    for i, item in enumerate(items):
        tier = tier_from_id(item.id)
        q_type = item.question["type"]

        t0 = time.monotonic()
        try:
            answer = model.predict(item.state, item.question)
        except Exception as e:
            print(f"  WARN: {item.id} failed: {e}", file=sys.stderr)
            all_failed += 1
            continue
        elapsed = time.monotonic() - t0
        latencies.append(elapsed)

        correct = False
        if q_type == "noul":
            noul_val = answer.get("noul", 0.5)
            predicted = noul_val > 0.5
            correct = predicted == item.label
        elif q_type == "choice":
            predicted = answer.get("choice", "")
            correct = predicted == item.label
        elif q_type == "score":
            probs = answer.get("probabilities", {})
            predicted = max(probs, key=probs.get) if probs else "0"
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

    summary = {
        "overall": {
            "accuracy": all_correct / all_total if all_total else 0,
            "correct": all_correct,
            "total": all_total,
            "failed": all_failed,
        },
        "by_tier": {},
        "by_type": {},
    }

    for tier, data in sorted(results_by_tier.items()):
        summary["by_tier"][tier] = {
            "accuracy": data["correct"] / data["total"] if data["total"] else 0,
            "correct": data["correct"],
            "total": data["total"],
        }

    for qtype, data in sorted(results_by_type.items()):
        summary["by_type"][qtype] = {
            "accuracy": data["correct"] / data["total"] if data["total"] else 0,
            "correct": data["correct"],
            "total": data["total"],
        }

    if latencies:
        latencies.sort()
        summary["latency"] = {
            "mean_s": sum(latencies) / len(latencies),
            "median_s": latencies[len(latencies) // 2],
            "total_s": sum(latencies),
        }

    return summary


def print_results(repo_id: str, results: dict) -> None:
    name = repo_id.split("/")[-1]
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")

    ov = results["overall"]
    print(f"Overall: {ov['accuracy']:.1%} ({ov['correct']}/{ov['total']})")
    if ov["failed"]:
        print(f"  Failed: {ov['failed']}")

    print("\nBy tier:")
    for tier, data in results["by_tier"].items():
        print(f"  {tier:10s}: {data['accuracy']:6.1%} ({data['correct']}/{data['total']})")

    print("\nBy type:")
    for qtype, data in results["by_type"].items():
        print(f"  {qtype:10s}: {data['accuracy']:6.1%} ({data['correct']}/{data['total']})")

    if "latency" in results:
        lat = results["latency"]
        print(f"\nLatency: mean={lat['mean_s']:.2f}s median={lat['median_s']:.2f}s total={lat['total_s']:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Krino models on JevBench")
    parser.add_argument("--repo", type=str, help="HuggingFace repo ID")
    parser.add_argument("--all", action="store_true", help="Evaluate all Krino models")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for results JSON")
    parser.add_argument("--max-items", type=int, default=None, help="Max items to evaluate (for testing)")
    args = parser.parse_args()

    if not args.repo and not args.all:
        parser.error("Specify --repo or --all")

    repos = KRINO_REPOS if args.all else [args.repo]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading JevBench data...")
    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    print(f"  {len(items)} items loaded")

    all_results = {}
    for repo_id in repos:
        print(f"\n--- {repo_id} ---")
        try:
            model = load_krino_from_hf(repo_id, device=device)
        except Exception as e:
            print(f"  ERROR loading {repo_id}: {e}", file=sys.stderr)
            continue

        results = evaluate_jevbench(model, items)
        print_results(repo_id, results)
        all_results[repo_id] = results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for repo_id, res in all_results.items():
            name = repo_id.split("/")[-1]
            out_path = args.output / f"{name}.json"
            out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            print(f"  {out_path}")
        print(f"\nResults saved to {args.output}")

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON TABLE")
        print(f"{'=' * 70}")
        print(f"{'Model':40s} {'Overall':>8s} {'Easy':>8s} {'Hard':>8s} {'Orig':>8s}")
        print("-" * 70)
        for repo_id, res in all_results.items():
            name = repo_id.split("/")[-1]
            ov = res["overall"]["accuracy"]
            easy = res["by_tier"].get("easy", {}).get("accuracy", 0)
            hard = res["by_tier"].get("hard", {}).get("accuracy", 0)
            orig = res["by_tier"].get("original", {}).get("accuracy", 0)
            print(f"{name:40s} {ov:7.1%} {easy:7.1%} {hard:7.1%} {orig:7.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
