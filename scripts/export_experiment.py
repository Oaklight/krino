"""Export experiment results from history.json to standardized experiment JSON.

Usage:
    python scripts/export_experiment.py \
        --experiment-id r1-baseline \
        --name "Round 1: Multi-task Baseline" \
        --description "19 benchmarks, type-balanced sampling, 20 epochs" \
        --checkpoints model/checkpoints/ettin_150m model/checkpoints/qwen3_0.6b ... \
        --models "Ettin-150m" "Qwen3-0.6B" ... \
        --output web/public/results/experiments/r1-baseline.json

    python scripts/export_experiment.py \
        --experiment-id head-sweep \
        --name "Head Architecture Sweep" \
        --description "5 epochs, rank x mlp_layers grid search" \
        --sweep-dir model/checkpoints \
        --sweep-pattern "sweep_*" \
        --output web/public/results/experiments/head-sweep.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def load_history(checkpoint_dir: Path) -> list[dict]:
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        return []
    with history_path.open() as f:
        return json.load(f)


def extract_run_info(history: list[dict], model_name: str, run_id: str) -> dict:
    """Extract standardized run info from a history.json."""
    epochs = []
    for entry in history:
        epoch_data = {
            "epoch": entry["epoch"],
            "train_loss": entry["train"]["mean_loss"],
            "elapsed_s": entry.get("elapsed_s", 0),
            "lr": entry.get("lr", 0),
        }
        if "eval" in entry:
            ev = entry["eval"]
            epoch_data["eval_loss"] = ev["aggregate"]["mean_loss"]
            epoch_data["eval_accuracy"] = ev["aggregate"]["accuracy"]
            epoch_data["by_type"] = {
                t: {"accuracy": v["accuracy"], "loss": v["mean_loss"], "n": v["n_items"]}
                for t, v in ev.get("by_type", {}).items()
            }
            epoch_data["by_source"] = {
                s: {
                    "accuracy": v["accuracy"],
                    "loss": v["mean_loss"],
                    "n": v["n_items"],
                    "type": v.get("type", "unknown"),
                }
                for s, v in ev.get("by_source", {}).items()
            }
        epochs.append(epoch_data)

    eval_epochs = [e for e in epochs if "eval_accuracy" in e]
    best = max(eval_epochs, key=lambda e: e["eval_accuracy"]) if eval_epochs else None

    return {
        "run_id": run_id,
        "model": model_name,
        "n_epochs": len(epochs),
        "best_epoch": best["epoch"] if best else None,
        "best_accuracy": best["eval_accuracy"] if best else None,
        "best_loss": best["eval_loss"] if best else None,
        "best_by_type": best.get("by_type") if best else None,
        "best_by_source": best.get("by_source") if best else None,
        "epochs": epochs,
    }


def parse_sweep_dir_name(dirname: str) -> dict | None:
    """Parse sweep_{model}_r{rank}_mlp{layers} into metadata."""
    m = re.match(r"sweep_(.+)_r(\d+)_mlp(\d+)", dirname)
    if not m:
        return None
    return {
        "model_key": m.group(1),
        "rank": int(m.group(2)),
        "mlp_layers": int(m.group(3)),
    }


def export_standard(args) -> dict:
    """Export from explicit checkpoint dirs + model names."""
    runs = []
    for ckpt_dir, model_name in zip(args.checkpoints, args.models):
        ckpt_path = Path(ckpt_dir)
        history = load_history(ckpt_path)
        if not history:
            print(f"  Warning: no history.json in {ckpt_dir}")
            continue
        run_id = f"{args.experiment_id}_{ckpt_path.name}"
        runs.append(extract_run_info(history, model_name, run_id))
        print(f"  {model_name}: {len(history)} epochs, best={runs[-1]['best_accuracy']:.4f}")

    return {
        "experiment_id": args.experiment_id,
        "name": args.name,
        "description": args.description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": runs,
    }


def export_sweep(args) -> dict:
    """Export from sweep directory pattern."""
    sweep_base = Path(args.sweep_dir)
    sweep_dirs = sorted(sweep_base.glob(args.sweep_pattern))

    runs = []
    for d in sweep_dirs:
        if not d.is_dir():
            continue
        meta = parse_sweep_dir_name(d.name)
        if not meta:
            continue
        history = load_history(d)
        if not history:
            continue
        model_name = f"{meta['model_key']} r={meta['rank']} mlp={meta['mlp_layers']}"
        run_id = d.name
        run_info = extract_run_info(history, model_name, run_id)
        run_info["config"] = meta
        runs.append(run_info)
        best = run_info["best_accuracy"]
        print(f"  {model_name}: {len(history)} epochs, best={best:.4f}" if best else f"  {model_name}: no eval")

    return {
        "experiment_id": args.experiment_id,
        "name": args.name,
        "description": args.description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "sweep_grid": {
            "ranks": sorted(set(r["config"]["rank"] for r in runs)),
            "mlp_layers": sorted(set(r["config"]["mlp_layers"] for r in runs)),
            "models": sorted(set(r["config"]["model_key"] for r in runs)),
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Export experiment results to JSON")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--output", type=Path, required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints", nargs="+", help="Checkpoint directories")
    group.add_argument("--sweep-dir", help="Base directory for sweep checkpoints")

    parser.add_argument("--models", nargs="+", help="Model names (for --checkpoints)")
    parser.add_argument("--sweep-pattern", default="sweep_*", help="Glob pattern for sweep dirs")

    args = parser.parse_args()

    if args.checkpoints:
        if not args.models or len(args.models) != len(args.checkpoints):
            parser.error("--models must match --checkpoints count")
        result = export_standard(args)
    else:
        result = export_sweep(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExported {result['n_runs']} runs to {args.output}")


if __name__ == "__main__":
    main()
