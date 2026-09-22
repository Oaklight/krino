#!/usr/bin/env python3
"""Evaluate the Jev API on benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probing" / "scripts"))

from jev_client import JevClient
from data.pipeline import load_jsonl
from model.evaluation.accuracy import noul_accuracy, choice_accuracy, score_mae
from model.evaluation.calibration import expected_calibration_error, brier_score


def load_env(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, ignoring comments and blank lines."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("'\"")
    return env


def evaluate_dataset(
    client: JevClient,
    dataset_path: Path,
    max_items: int | None = None,
    delay: float = 0.1,
) -> dict:
    items = load_jsonl(dataset_path)
    if max_items:
        items = items[:max_items]

    noul_preds, noul_labels = [], []
    choice_preds, choice_labels = [], []
    score_preds, score_labels = [], []
    noul_confs, noul_correct = [], []
    choice_confs, choice_correct = [], []
    ttfb_times, total_times = [], []
    n_errors = 0

    for i, item in enumerate(items):
        try:
            raw = client.ask_raw(item.state, {"q": item.question})
            ttfb_times.append(raw.ttfb_s)
            total_times.append(raw.total_s)

            if raw.status != 200:
                msg = raw.body.decode("utf-8", errors="replace")[:200]
                print(f"  [{i}] HTTP {raw.status}: {msg}")
                n_errors += 1
                if delay > 0:
                    time.sleep(delay)
                continue

            parsed = raw.json()
            answer = parsed["answers"]["q"]
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            n_errors += 1
            if delay > 0:
                time.sleep(delay)
            continue

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

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)} (ttfb={raw.ttfb_s:.3f}s total={raw.total_s:.3f}s)")

        if delay > 0:
            time.sleep(delay)

    results: dict = {"items": len(items)}
    if n_errors:
        results["errors"] = n_errors

    if noul_preds:
        results["noul"] = {
            "accuracy": noul_accuracy(noul_preds, noul_labels),
            "ece": expected_calibration_error(noul_confs, noul_correct),
            "brier": brier_score(noul_preds, [1.0 if lb else 0.0 for lb in noul_labels]),
        }

    if choice_preds:
        results["choice"] = {
            "accuracy": choice_accuracy(choice_preds, choice_labels),
            "ece": expected_calibration_error(choice_confs, choice_correct),
        }

    if score_preds:
        results["score"] = score_mae(score_preds, score_labels)

    if ttfb_times:
        ttfb_times.sort()
        total_times.sort()
        n = len(ttfb_times)
        results["latency"] = {
            "ttfb_mean_s": sum(ttfb_times) / n,
            "ttfb_median_s": ttfb_times[n // 2],
            "ttfb_p95_s": ttfb_times[int(n * 0.95)],
            "total_mean_s": sum(total_times) / n,
            "total_median_s": total_times[n // 2],
            "total_p95_s": total_times[int(n * 0.95)],
            "total_sum_s": sum(total_times),
            "n_requests": n,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Jev API on benchmarks")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset JSONL paths")
    parser.add_argument("--max-items", type=int, default=200, help="Max items per dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    parser.add_argument("--timeout", type=float, default=30.0, help="API request timeout (seconds)")
    args = parser.parse_args()

    env = load_env(ROOT / ".env")
    api_key = env.get("TYPESAFE_API_KEY")
    base_url = env.get("TYPESAFE_BASE_URL")
    if not api_key:
        print("Error: TYPESAFE_API_KEY not found in .env")
        return 1

    client = JevClient(api_key=api_key, base_url=base_url, timeout=args.timeout)
    print(f"Jev API: {client.base_url}")

    dataset_paths = args.datasets
    if not dataset_paths:
        benchmarks = ROOT / "model" / "data" / "benchmarks"
        dataset_paths = sorted(
            str(p) for p in benchmarks.rglob("*.jsonl") if not p.stem.endswith("_raw")
        )
        if not dataset_paths:
            print("No datasets found. Run data pipeline first: python -m data.pipeline")
            return 1

    all_results = {}
    for ds_path in dataset_paths:
        path = Path(ds_path)
        name = path.stem
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")

        results = evaluate_dataset(client, path, max_items=args.max_items, delay=args.delay)

        for section, data in results.items():
            if not isinstance(data, dict):
                continue
            if "accuracy" in data:
                acc = data["accuracy"]
                if isinstance(acc, dict):
                    print(f"  {section} accuracy: {acc.get('accuracy', 'N/A'):.4f} (n={acc.get('n', 0)})")
                ece = data.get("ece", {})
                if isinstance(ece, dict) and "ece" in ece:
                    print(f"  {section} ECE: {ece['ece']:.4f}")
            elif "mae" in data:
                print(f"  {section} MAE: {data['mae']:.4f} (n={data.get('n', 0)})")

        lat = results.get("latency", {})
        if lat:
            print(f"  latency: ttfb_mean={lat['ttfb_mean_s']:.3f}s total_mean={lat['total_mean_s']:.3f}s total_p95={lat['total_p95_s']:.3f}s")

        all_results[name] = results

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
