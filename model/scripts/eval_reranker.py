#!/usr/bin/env python3
"""Evaluate reranker-based scoring baseline on benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.reranker_scorer import RerankerScorer, load_reranker
from model.data.pipeline import load_jsonl
from model.evaluation.accuracy import noul_accuracy, choice_accuracy, score_mae
from model.evaluation.calibration import expected_calibration_error, brier_score


def evaluate_dataset(scorer, dataset_path, max_items=None):
    items = load_jsonl(dataset_path)
    if max_items:
        items = items[:max_items]

    noul_preds, noul_labels = [], []
    choice_preds, choice_labels = [], []
    score_preds, score_labels = [], []
    noul_confs, noul_correct = [], []
    choice_confs, choice_correct = [], []
    latencies = []

    for i, item in enumerate(items):
        t0 = time.monotonic()
        answer = scorer.predict(item.state, item.question)
        elapsed = time.monotonic() - t0
        latencies.append(elapsed)

        q_type = item.question["type"]
        if q_type == "noul":
            noul_val = answer.get("noul", 0.5)
            label = item.label if isinstance(item.label, bool) else str(item.label).lower() == "true"
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
        elif q_type == "score":
            score_preds.append(answer.get("score", 0.0))
            score_labels.append(float(item.label))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)} ({elapsed:.3f}s/item)")

    results = {"items": len(items)}
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
    if score_preds:
        results["score"] = score_mae(score_preds, score_labels)
    if latencies:
        latencies.sort()
        results["latency"] = {
            "mean_s": sum(latencies) / len(latencies),
            "median_s": latencies[len(latencies) // 2],
            "p95_s": latencies[int(len(latencies) * 0.95)],
            "total_s": sum(latencies),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker scoring baseline")
    parser.add_argument("--model", default="cross-encoder/ettin-reranker-150m-v1", help="Reranker model name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (use float16 for V100)")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = load_reranker(args.model, device=args.device, dtype=args.dtype)
    scorer = RerankerScorer(model)
    print(f"Model loaded on {model.device}")

    dataset_paths = args.datasets
    if not dataset_paths:
        benchmarks = ROOT / "model" / "data" / "benchmarks"
        dataset_paths = sorted(str(p) for p in benchmarks.rglob("*.jsonl"))

    all_results = {}
    for ds_path in dataset_paths:
        path = Path(ds_path)
        name = path.stem
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        results = evaluate_dataset(scorer, path, max_items=args.max_items)
        for section, data in results.items():
            if not isinstance(data, dict):
                continue
            if "accuracy" in data:
                acc = data["accuracy"]
                if isinstance(acc, dict):
                    print(f"  {section} accuracy: {acc.get('accuracy', 'N/A'):.4f} (n={acc.get('n', 0)})")
                ece = data.get("ece", {})
                if isinstance(ece, dict):
                    print(f"  {section} ECE: {ece.get('ece', 'N/A'):.4f}")
            elif "mae" in data:
                print(f"  {section} MAE: {data['mae']:.4f} (n={data.get('n', 0)})")
        if "latency" in results:
            lat = results["latency"]
            print(f"  latency: mean={lat['mean_s']:.3f}s median={lat['median_s']:.3f}s p95={lat['p95_s']:.3f}s")
        all_results[name] = results

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
