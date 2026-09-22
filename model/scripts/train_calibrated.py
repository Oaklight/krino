#!/usr/bin/env python3
"""Train decision heads with calibration-aware losses.

Extends the supervised training script with configurable loss functions
(Brier, MMCE, focal, label smoothing) via YAML config or CLI args.

Usage:
    # Single run with CLI args:
    python -m model.scripts.train_calibrated \
        --model Qwen/Qwen3-0.6B \
        --train-data model/data/benchmarks/banking77.jsonl \
        --eval-data model/data/benchmarks/banking77.jsonl \
        --loss brier --alpha 0.7 --beta 0.3

    # Sweep from YAML config:
    python -m model.scripts.train_calibrated \
        --config model/configs/qwen_0.6b_calibrated.yaml \
        --run brier_07
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jsonl
from model.src.backbone import load_causal_lm, load_encoder
from model.src.decision_model import DecisionModel
from model.training.calibration import CalibrationLossConfig, train


def load_config(path: Path, run_name: str | None = None) -> dict:
    """Load YAML config, optionally selecting a specific run."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})

    if run_name:
        runs = cfg.get("runs", {})
        if run_name not in runs:
            available = list(runs.keys())
            print(f"Error: run '{run_name}' not found. Available: {available}", file=sys.stderr)
            sys.exit(1)
        run_cfg = runs[run_name]
        merged = {**defaults, **run_cfg}
        merged["run_name"] = run_name
        return merged

    return defaults


def build_loss_config(cfg: dict) -> CalibrationLossConfig:
    """Build CalibrationLossConfig from a flat dict."""
    loss_fields = {}
    for key in CalibrationLossConfig.__dataclass_fields__:
        if key in cfg:
            loss_fields[key] = cfg[key]

    # "loss" key maps to "name"
    if "loss" in cfg and "name" not in loss_fields:
        loss_fields["name"] = cfg["loss"]

    return CalibrationLossConfig.from_dict(loss_fields)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train with calibration losses")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--run", type=str, help="Run name from config")
    parser.add_argument("--model", default=None, help="Backbone model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-data", nargs="+", default=None)
    parser.add_argument("--eval-data", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--rival-aware", action="store_true", default=None)
    parser.add_argument("--encoder", action="store_true", default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Loss config CLI overrides
    parser.add_argument("--loss", type=str, default=None, choices=["brier", "mmce", "focal"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mmce-bandwidth", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        cfg = load_config(args.config, args.run)
    else:
        cfg = {}

    # CLI args override config
    def resolve(cli_val, cfg_key, default=None):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, default)

    model_name = resolve(args.model, "model", "Qwen/Qwen3-0.6B")
    device = resolve(args.device, "device")
    train_paths = args.train_data or cfg.get("train_data", [])
    eval_paths = args.eval_data or cfg.get("eval_data", [])
    epochs = int(resolve(args.epochs, "epochs", 10))
    lr = float(resolve(args.lr, "lr", 1e-3))
    rank = resolve(args.rank, "rank", 64)
    rival_aware = resolve(args.rival_aware, "rival_aware", False)
    use_encoder = resolve(args.encoder, "encoder", False)
    max_train = resolve(args.max_train, "max_train")
    max_eval = resolve(args.max_eval, "max_eval")
    checkpoint_dir = args.checkpoint_dir or cfg.get("checkpoint_dir")
    output = args.output or cfg.get("output")
    seed = resolve(args.seed, "seed", 42)
    run_name = cfg.get("run_name", "default")

    # Build loss config: start from config, override with CLI
    loss_cfg = {}
    for key in ["loss", "name", "alpha", "beta", "gamma", "label_smoothing", "mmce_bandwidth", "batch_size"]:
        if key in cfg:
            loss_cfg[key] = cfg[key]
    # CLI overrides
    if args.loss is not None:
        loss_cfg["loss"] = args.loss
    if args.alpha is not None:
        loss_cfg["alpha"] = args.alpha
    if args.beta is not None:
        loss_cfg["beta"] = args.beta
    if args.gamma is not None:
        loss_cfg["gamma"] = args.gamma
    if args.label_smoothing is not None:
        loss_cfg["label_smoothing"] = args.label_smoothing
    if args.mmce_bandwidth is not None:
        loss_cfg["mmce_bandwidth"] = args.mmce_bandwidth
    if args.batch_size is not None:
        loss_cfg["batch_size"] = args.batch_size

    cal_cfg = build_loss_config(loss_cfg)

    if not train_paths or not eval_paths:
        print("Error: --train-data and --eval-data required (or set in config)", file=sys.stderr)
        return 1

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"=== Calibration training: {run_name} ===")
    print(f"Backbone: {model_name} ({'encoder' if use_encoder else 'causal'})")

    if use_encoder:
        backbone, tokenizer = load_encoder(model_name, device=device, freeze=True)
    else:
        backbone, tokenizer = load_causal_lm(model_name, device=device, freeze=True)

    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=rank,
        rival_aware=rival_aware,
    )
    print(f"Trainable: {model.trainable_parameters():,} params")
    print(f"Frozen:    {model.frozen_parameters():,} params")

    train_items = []
    for path in train_paths:
        train_items.extend(load_jsonl(Path(path)))
    if max_train:
        random.shuffle(train_items)
        train_items = train_items[:max_train]

    eval_items = []
    for path in eval_paths:
        eval_items.extend(load_jsonl(Path(path)))
    if max_eval:
        eval_items = eval_items[:max_eval]

    supported = {"noul", "choice", "score"}
    train_items = [it for it in train_items if it.question.get("type") in supported]
    eval_items = [it for it in eval_items if it.question.get("type") in supported]
    print(f"Train: {len(train_items)}, Eval: {len(eval_items)}")

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = train(
        model=model,
        train_items=train_items,
        eval_items=eval_items,
        loss_cfg=cal_cfg,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
    )

    final_eval = next(
        (e["eval"] for e in reversed(results["history"]) if "eval" in e), {}
    )
    loss_str = f"{final_eval['mean_loss']:.4f}" if "mean_loss" in final_eval else "N/A"
    acc_str = f"{final_eval['accuracy']:.4f}" if "accuracy" in final_eval else "N/A"
    print(f"\nFinal: loss={loss_str} acc={acc_str}")

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
