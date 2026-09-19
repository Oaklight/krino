#!/usr/bin/env python3
"""Evaluate trained decision heads on multiple benchmarks.

Loads a backbone + saved head checkpoint and evaluates on one or more
benchmark JSONL files. Used for cross-benchmark generalization testing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import load_causal_lm, load_encoder
from model.src.decision_model import DecisionModel
from model.data.pipeline import load_jsonl
from model.evaluation.accuracy import noul_accuracy, choice_accuracy
from model.evaluation.calibration import expected_calibration_error, brier_score


def evaluate_items(model, items):
    noul_preds, noul_labels = [], []
    choice_preds, choice_labels = [], []
    noul_confs, noul_correct = [], []
    choice_confs, choice_correct = [], []
    latencies = []
    n_unsupported = 0
    n_failed = 0

    for i, item in enumerate(items):
        q_type = item.question.get("type", "")
        if q_type not in ("noul", "choice"):
            n_unsupported += 1
            continue

        t0 = time.monotonic()
        try:
            answer = model.predict(item.state, item.question)
        except Exception as e:
            print(f"    WARN: item {i} failed: {e}", file=sys.stderr)
            n_failed += 1
            continue
        latencies.append(time.monotonic() - t0)

        if q_type == "noul":
            noul_val = answer.get("noul", 0.5)
            label = item.label
            if isinstance(label, str):
                label = label.lower() in ("true", "yes", "1")
            noul_preds.append(noul_val)
            noul_labels.append(label)
            conf = noul_val if label else 1 - noul_val
            noul_confs.append(conf)
            noul_correct.append((noul_val > 0.5) == label)
        elif q_type == "choice":
            choice_val = answer.get("choice", "")
            choice_preds.append(choice_val)
            choice_labels.append(item.label)
            probs = answer.get("probabilities", {})
            choice_confs.append(probs.get(choice_val, 0))
            choice_correct.append(choice_val == item.label)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(items)}")

    results = {"items_evaluated": len(items) - n_unsupported - n_failed, "items_unsupported": n_unsupported, "items_failed": n_failed}
    if noul_preds:
        results["noul"] = {
            "accuracy": noul_accuracy(noul_preds, noul_labels),
            "ece": expected_calibration_error(noul_confs, noul_correct),
            "brier": brier_score(noul_preds, [1.0 if l else 0.0 for l in noul_labels]),
        }
    if choice_preds:
        results["choice"] = {
            "accuracy": choice_accuracy(choice_preds, choice_labels),
            "ece": expected_calibration_error(choice_confs, choice_correct),
        }
    if latencies:
        latencies.sort()
        results["latency"] = {
            "mean_s": sum(latencies) / len(latencies),
            "median_s": latencies[len(latencies) // 2],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained heads on benchmarks")
    parser.add_argument("--model", required=True, help="Backbone model name")
    parser.add_argument("--encoder", action="store_true")
    parser.add_argument("--heads", type=Path, required=True, help="Path to saved heads checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rival-aware", action="store_true")
    parser.add_argument("--datasets", nargs="+", required=True, help="Eval JSONL files")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading backbone: {args.model}")
    if args.encoder:
        backbone, tokenizer = load_encoder(args.model, device=args.device, freeze=True)
    else:
        backbone, tokenizer = load_causal_lm(args.model, device=args.device, freeze=True)

    model = DecisionModel(
        backbone=backbone, tokenizer=tokenizer,
        rank=args.rank, rival_aware=args.rival_aware,
    )
    model.load_heads(args.heads)
    model.eval()
    print(f"Heads loaded from {args.heads}")

    all_results = {}
    for ds_path in args.datasets:
        path = Path(ds_path)
        name = path.stem
        items = load_jsonl(path)
        if args.max_items:
            items = items[:args.max_items]

        print(f"\n  Evaluating: {name} ({len(items)} items)")
        results = evaluate_items(model, items)

        for section in ("noul", "choice"):
            if section in results:
                acc = results[section].get("accuracy", {})
                if isinstance(acc, dict):
                    print(f"    {section} accuracy: {acc.get('accuracy', 'N/A'):.4f} (n={acc.get('n', 0)})")

        all_results[name] = results

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
