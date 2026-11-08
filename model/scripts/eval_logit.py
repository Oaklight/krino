#!/usr/bin/env python3
"""Evaluate the logit readout baseline on benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import is_hybrid_model, load_causal_lm
from model.src.logit_scorer import LogitScorer
from data.pipeline import load_jsonl
from model.evaluation.accuracy import noul_accuracy, choice_accuracy, score_mae
from model.evaluation.calibration import expected_calibration_error, brier_score


def evaluate_dataset(
    scorer: LogitScorer,
    dataset_path: Path,
    max_items: int | None = None,
) -> dict:
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
            label = (
                item.label
                if isinstance(item.label, bool)
                else str(item.label).lower() == "true"
            )
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


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    model_id: str,
    revision: str | None,
    dtype: str,
    device: str,
    strategy: str,
    norm: str,
    max_items: int | None,
    dataset_paths: list[str],
    license_name: str | None,
    variant: str | None = None,
    readout: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> dict:
    """The record a reader needs to re-run this file and get these numbers.

    The Phase A files carry none of this, which is why they cannot be audited.
    ``snapshot_path`` is relative to the Hugging Face cache root so it names the
    same location on any machine without carrying a home directory, and the
    dataset paths are repo-relative for the same reason.
    """
    dirty = _git(["status", "--porcelain"]) not in (None, "")
    snapshot = None
    if revision and "/" in model_id:
        org, name = model_id.split("/", 1)
        snapshot = f"models--{org}--{name}/snapshots/{revision}"
    datasets = {}
    for raw in dataset_paths:
        path = Path(raw)
        try:
            shown = str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            # Outside the checkout: record the name, never a host path.
            shown = path.name
        datasets[path.stem] = {"path": shown, "sha256": _sha256(path)}
    return {
        "model_id": model_id,
        "revision": revision,
        "revision_is_commit": bool(revision and len(revision) == 40),
        "variant": variant,
        "readout": readout or strategy,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "snapshot_path": snapshot,
        "license": license_name,
        "dtype": dtype,
        "device": device,
        "strategy": strategy,
        "norm": norm,
        "max_items": max_items,
        "code_commit": _git(["rev-parse", "HEAD"]),
        "code_dirty": dirty,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate logit readout baseline")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="Dataset JSONL paths"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Max items per dataset"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument(
        "--norm", default="mean", choices=["mean", "sum"], help="Log-prob normalization"
    )
    parser.add_argument(
        "--strategy",
        default="description",
        choices=["description", "label"],
        help="Choice scoring strategy",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (use float16 for V100)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Commit, tag or branch to load and record; pass a commit SHA to pin",
    )
    parser.add_argument(
        "--license",
        default=None,
        help="Licence identifier recorded in the provenance block",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["base", "instruct", "reranker", "unrecorded"],
        help="Checkpoint flavour, recorded so Base and Instruct rows are not pooled",
    )
    parser.add_argument(
        "--readout",
        default=None,
        help="Readout name recorded in provenance; defaults to the strategy",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"Loading {args.model}...")
    model, tokenizer = load_causal_lm(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        revision=args.revision,
    )

    # Hybrid models (GDN, e.g. Qwen3.5) don't support KV-cache batching —
    # force sequential scoring (batch_size=1) to avoid cache shape mismatches.
    hybrid = is_hybrid_model(model.config)
    batch_size = 1 if hybrid else 64
    if hybrid:
        print("Hybrid model detected — using sequential scoring (no KV cache batching)")

    scorer = LogitScorer(
        model, tokenizer, norm=args.norm, strategy=args.strategy, batch_size=batch_size
    )
    print(f"Strategy: {args.strategy}, norm: {args.norm}")
    print(f"Model loaded on {next(model.parameters()).device}")

    dataset_paths = args.datasets
    if not dataset_paths:
        benchmarks = ROOT / "model" / "data" / "benchmarks"
        dataset_paths = sorted(str(p) for p in benchmarks.rglob("*.jsonl"))
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

        results = evaluate_dataset(scorer, path, max_items=args.max_items)

        for section, data in results.items():
            if not isinstance(data, dict):
                continue
            if "accuracy" in data:
                acc = data["accuracy"]
                if isinstance(acc, dict):
                    print(
                        f"  {section} accuracy: {acc.get('accuracy', 'N/A'):.4f} (n={acc.get('n', 0)})"
                    )
                ece = data.get("ece", {})
                if isinstance(ece, dict):
                    print(f"  {section} ECE: {ece.get('ece', 'N/A'):.4f}")
            elif "mae" in data:
                print(f"  {section} MAE: {data['mae']:.4f} (n={data.get('n', 0)})")

        if "latency" in results:
            lat = results["latency"]
            print(
                f"  latency: mean={lat['mean_s']:.3f}s median={lat['median_s']:.3f}s p95={lat['p95_s']:.3f}s"
            )

        all_results[name] = results

    if args.output:
        provenance = build_provenance(
            model_id=args.model,
            revision=args.revision,
            dtype=args.dtype,
            device=str(next(model.parameters()).device),
            strategy=args.strategy,
            norm=args.norm,
            max_items=args.max_items,
            dataset_paths=[str(Path(p)) for p in dataset_paths],
            license_name=args.license,
            variant=args.variant,
            readout=args.readout,
            tokenizer_id=getattr(tokenizer, "name_or_path", None),
            tokenizer_revision=(getattr(tokenizer, "init_kwargs", None) or {}).get(
                "_commit_hash"
            ),
        )
        provenance["benchmarks_scored"] = len(all_results)
        provenance["items_per_benchmark"] = {
            name: value.get("items") for name, value in all_results.items()
        }
        payload = {"provenance": provenance, **all_results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
