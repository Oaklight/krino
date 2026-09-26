#!/usr/bin/env python3
"""Multi-task training of decision heads across multiple benchmarks.

Trains type-balanced decision heads on 19+ benchmarks with per-source
weighting, capping, and detailed per-type/per-source evaluation.

Usage:
    # From YAML config:
    python model/scripts/train_multitask.py --config model/configs/multitask_ettin150m.yaml

    # With CLI overrides:
    python model/scripts/train_multitask.py \
        --config model/configs/multitask_ettin150m.yaml \
        --epochs 5 --lr 0.0005 --device cuda:1

    # Eval-only (skip training, just evaluate with loaded heads):
    python model/scripts/train_multitask.py \
        --config model/configs/multitask_ettin150m.yaml \
        --eval-only --load-heads model/experiments/best_heads.pt

    # GPU memory overrides:
    python model/scripts/train_multitask.py \
        --config model/configs/multitask_ettin150m.yaml \
        --no-flash-attention --bf16 --max-length 4096
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

from data.format import TypedQuestion
from data.pipeline import load_jsonl
from data.sampler import SamplerConfig
from model.src.backbone import load_causal_lm, load_encoder, load_qwen35_base
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info
from model.training.supervised import eval_epoch_detailed, train_multitask


def load_config(path: Path) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def load_source_items(
    source_name: str,
    source_cfg: dict,
    rng: random.Random,
) -> tuple[list[TypedQuestion], list[TypedQuestion]]:
    """Load and split items for a single source.

    Args:
        source_name: Name of the source (e.g. "banking77").
        source_cfg: Source config dict with keys: path, weight, max_train, max_eval.
        rng: Random instance for deterministic sampling.

    Returns:
        (train_items, eval_items) tuple.
    """
    path_str = source_cfg.get("path")
    if not path_str:
        # Default path convention
        path_str = f"model/data/benchmarks/{source_name}.jsonl"
    path = resolve_path(path_str)

    if not path.exists():
        print(
            f"  WARNING: {path} not found, skipping {source_name}",
            file=sys.stderr,
            flush=True,
        )
        return [], []

    items = load_jsonl(path)

    # Filter to supported types
    supported = {"noul", "choice", "score"}
    items = [it for it in items if it.question.get("type") in supported]

    # Split by item.split field
    train_items = [it for it in items if it.split == "train"]
    eval_items = [it for it in items if it.split == "test"]

    # Apply max_train / max_eval caps
    max_train = source_cfg.get("max_train")
    if max_train is not None and len(train_items) > max_train:
        rng.shuffle(train_items)
        train_items = train_items[:max_train]

    max_eval = source_cfg.get("max_eval")
    if max_eval is not None and len(eval_items) > max_eval:
        rng.shuffle(eval_items)
        eval_items = eval_items[:max_eval]

    # Sources with weight: 0 are eval-only
    weight = source_cfg.get("weight", 1.0)
    if weight == 0:
        train_items = []

    return train_items, eval_items


def build_sampler_config(cfg: dict, sources_cfg: dict) -> SamplerConfig:
    """Build SamplerConfig from the config dict.

    Merges top-level sampling config with per-source weights and caps.
    """
    sampling = cfg.get("sampling", {})

    type_ratios = sampling.get(
        "type_ratios", {"noul": 1.0, "choice": 1.0, "score": 1.0}
    )
    difficulty_weights = sampling.get("difficulty_weights")
    epoch_size = sampling.get("epoch_size")
    sampling_temperature = sampling.get("sampling_temperature")
    accumulation_steps = cfg.get("accumulation_steps", 8)

    source_weights: dict[str, float] = {}
    for source_name, source_cfg in sources_cfg.items():
        weight = source_cfg.get("weight", 1.0)
        if weight > 0:
            source_weights[source_name] = weight

    # No source_caps — capping already done in load_source_items
    return SamplerConfig(
        type_ratios=type_ratios,
        difficulty_weights=difficulty_weights,
        source_weights=source_weights,
        source_caps={},
        epoch_size=epoch_size,
        accumulation_steps=accumulation_steps,
        sampling_temperature=sampling_temperature,
    )


def print_data_summary(
    train_items: list[TypedQuestion],
    eval_items: list[TypedQuestion],
) -> None:
    """Print a summary of loaded data by source and type."""
    from collections import Counter

    train_by_source: Counter[str] = Counter()
    train_by_type: Counter[str] = Counter()
    eval_by_source: Counter[str] = Counter()

    for item in train_items:
        train_by_source[item.source] += 1
        train_by_type[item.question.get("type", "?")] += 1
    for item in eval_items:
        eval_by_source[item.source] += 1

    print("\n  Data summary:", flush=True)
    print(f"  {'Source':<20} {'Train':>8} {'Eval':>8}", flush=True)
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8}", flush=True)
    all_sources = sorted(
        set(list(train_by_source.keys()) + list(eval_by_source.keys()))
    )
    for source in all_sources:
        t = train_by_source.get(source, 0)
        e = eval_by_source.get(source, 0)
        print(f"  {source:<20} {t:>8,} {e:>8,}", flush=True)
    print(f"  {'TOTAL':<20} {len(train_items):>8,} {len(eval_items):>8,}", flush=True)

    print(f"\n  Type distribution (train): {dict(train_by_type)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-task decision head training")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--model", default=None, help="Override backbone model")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="Checkpoint directory"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output results JSON")
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training, just evaluate"
    )
    parser.add_argument(
        "--load-heads", type=Path, default=None, help="Load heads from checkpoint"
    )
    parser.add_argument("--rank", type=int, default=None, help="Override attention rank")
    parser.add_argument("--mlp-layers", type=int, default=None, help="Override MLP projector layers")
    parser.add_argument("--noul-rank", type=int, default=None, help="Override NoulHead MLP rank")
    parser.add_argument("--max-length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--mlp-lr", type=float, default=None, help="Separate LR for MLP projector")
    parser.add_argument("--noul-lr", type=float, default=None, help="Separate LR for noul head")
    parser.add_argument("--choice-lr", type=float, default=None, help="Separate LR for choice head")
    parser.add_argument("--score-lr", type=float, default=None, help="Separate LR for score head")
    parser.add_argument("--uncertainty-weighting", action="store_true", default=None, help="Enable uncertainty-based loss weighting (Kendall et al. 2018)")
    parser.add_argument("--sampling-temperature", type=float, default=None, help="Temperature for size-based source weighting (mT5/PaLM style, typical 0.3-0.7)")
    parser.add_argument("--difficulty-weights", type=str, default=None, help="JSON string of per-type difficulty weights, e.g. '{\"noul\": 0.5, \"choice\": 1.0, \"score\": 2.0}'")
    parser.add_argument("--save-every-epoch", action="store_true", default=None, help="Save checkpoint at every eval epoch")
    parser.add_argument("--eval-every", type=int, default=None, help="Override eval frequency")
    parser.add_argument("--shared-attention", action="store_true", default=None, help="Share AttentionHead weights between choice and score heads")

    # GPU memory optimization flags
    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument(
        "--flash-attention", action="store_true", default=None,
        help="Enable Flash Attention 2 (override auto-detection)",
    )
    flash_group.add_argument(
        "--no-flash-attention", action="store_true", default=None,
        help="Disable Flash Attention 2",
    )

    bf16_group = parser.add_mutually_exclusive_group()
    bf16_group.add_argument(
        "--bf16", action="store_true", default=None,
        help="Enable bf16 autocast for backbone (override auto-detection)",
    )
    bf16_group.add_argument(
        "--no-bf16", action="store_true", default=None,
        help="Disable bf16 autocast for backbone",
    )

    gc_group = parser.add_mutually_exclusive_group()
    gc_group.add_argument(
        "--gradient-checkpointing", action="store_true", default=None,
        help="Enable gradient checkpointing on backbone (override auto-detection)",
    )
    gc_group.add_argument(
        "--no-gradient-checkpointing", action="store_true", default=None,
        help="Disable gradient checkpointing",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve values with CLI overrides
    model_name = args.model or cfg.get("model", "Qwen/Qwen3-0.6B")
    use_encoder = cfg.get("encoder", False)
    qwen35_base = cfg.get("qwen35_base", False)
    rank = args.rank if args.rank is not None else cfg.get("rank", 64)
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 20)
    lr = args.lr if args.lr is not None else cfg.get("lr", 1e-3)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device or cfg.get("device")
    accumulation_steps = cfg.get("accumulation_steps", 8)
    batch_backbone = cfg.get("batch_backbone", 16)
    save_every_epoch = args.save_every_epoch if args.save_every_epoch is not None else cfg.get("save_every_epoch", False)
    eval_every = args.eval_every if args.eval_every is not None else cfg.get("eval_every", 2)
    rival_aware = cfg.get("rival_aware", False)
    mlp_layers = args.mlp_layers if args.mlp_layers is not None else cfg.get("mlp_layers", 0)
    mlp_dim = cfg.get("mlp_dim", None)
    noul_rank = args.noul_rank if args.noul_rank is not None else cfg.get("noul_rank", None)
    max_length = args.max_length if args.max_length is not None else cfg.get("max_length", None)
    mlp_lr = args.mlp_lr if args.mlp_lr is not None else cfg.get("mlp_lr", None)
    noul_lr = args.noul_lr if args.noul_lr is not None else cfg.get("noul_lr", None)
    choice_lr = args.choice_lr if args.choice_lr is not None else cfg.get("choice_lr", None)
    score_lr = args.score_lr if args.score_lr is not None else cfg.get("score_lr", None)
    uncertainty_weighting = args.uncertainty_weighting if args.uncertainty_weighting is not None else cfg.get("uncertainty_weighting", False)
    shared_attention = args.shared_attention if args.shared_attention is not None else cfg.get("shared_attention", False)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and cfg.get("checkpoint_dir"):
        checkpoint_dir = Path(cfg["checkpoint_dir"])

    output = args.output
    if output is None and cfg.get("output"):
        output = Path(cfg["output"])

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== Multi-task training ===", flush=True)
    if qwen35_base:
        arch_label = "qwen35-base (GDN hybrid)"
    elif use_encoder:
        arch_label = "encoder"
    else:
        arch_label = "causal"
    print(f"Backbone: {model_name} ({arch_label})", flush=True)
    print(f"Config: {args.config}", flush=True)

    # Auto-detect GPU config, then apply CLI overrides
    model_billions = estimate_model_billions(model_name)
    gpu_config = GPUConfig.auto_detect(model_billions)

    # Resolve CLI boolean flags (mutually exclusive groups)
    flash_override: bool | None = None
    if args.flash_attention:
        flash_override = True
    elif args.no_flash_attention:
        flash_override = False

    bf16_override: bool | None = None
    if args.bf16:
        bf16_override = True
    elif args.no_bf16:
        bf16_override = False

    gc_override: bool | None = None
    if args.gradient_checkpointing:
        gc_override = True
    elif args.no_gradient_checkpointing:
        gc_override = False

    gpu_config = gpu_config.apply_overrides(
        flash_attention=flash_override,
        bf16=bf16_override,
        gradient_checkpointing=gc_override,
        max_length=max_length,
    )

    print(f"\nGPU optimization config (model ~{model_billions:.1f}B params):", flush=True)
    print_gpu_info(gpu_config)

    # Load data from all sources
    sources_cfg = cfg.get("sources", {})
    if not sources_cfg:
        print("Error: no sources defined in config", file=sys.stderr, flush=True)
        return 1

    rng = random.Random(seed)
    all_train: list[TypedQuestion] = []
    all_eval: list[TypedQuestion] = []

    for source_name, source_cfg in sources_cfg.items():
        if source_cfg is None:
            source_cfg = {}
        train_items, eval_items = load_source_items(source_name, source_cfg, rng)
        all_train.extend(train_items)
        all_eval.extend(eval_items)
        n_train = len(train_items)
        n_eval = len(eval_items)
        if n_train > 0 or n_eval > 0:
            print(f"  {source_name}: {n_train} train, {n_eval} eval", flush=True)

    print_data_summary(all_train, all_eval)

    # Load backbone and build model
    print(f"\nLoading backbone: {model_name}", flush=True)
    flash_att = gpu_config.flash_attention
    if qwen35_base:
        backbone, tokenizer = load_qwen35_base(
            model_name, device=device, freeze=True, flash_attention=flash_att,
        )
    elif use_encoder:
        backbone, tokenizer = load_encoder(
            model_name, device=device, freeze=True, flash_attention=flash_att,
        )
    else:
        backbone, tokenizer = load_causal_lm(
            model_name, device=device, freeze=True, flash_attention=flash_att,
        )

    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=rank,
        rival_aware=rival_aware,
        mlp_layers=mlp_layers,
        mlp_dim=mlp_dim,
        noul_rank=noul_rank,
        max_length=max_length,
        gpu_config=gpu_config,
        shared_attention=shared_attention,
    )
    print(f"Trainable: {model.trainable_parameters():,} params", flush=True)
    print(f"Frozen:    {model.frozen_parameters():,} params", flush=True)

    # Load heads if specified
    if args.load_heads:
        print(f"Loading heads from {args.load_heads}", flush=True)
        model.load_heads(args.load_heads)

    # Eval-only mode
    if args.eval_only:
        print("\n=== Eval-only mode ===", flush=True)
        eval_stats = eval_epoch_detailed(model, all_eval)
        _print_detailed_eval(eval_stats)

        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(eval_stats, indent=2, default=str))
            print(f"\nResults saved to {output}", flush=True)
        return 0

    # Apply CLI sampling_temperature override
    if args.sampling_temperature is not None:
        sampling_section = cfg.setdefault("sampling", {})
        sampling_section["sampling_temperature"] = args.sampling_temperature
    # Apply CLI difficulty_weights override to sampling config
    if args.difficulty_weights is not None:
        sampling_section = cfg.setdefault("sampling", {})
        sampling_section["difficulty_weights"] = json.loads(args.difficulty_weights)

    # Build sampler config
    sampler_cfg = build_sampler_config(cfg, sources_cfg)

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    results = train_multitask(
        model=model,
        train_items=all_train,
        eval_items=all_eval,
        sampler_config=sampler_cfg,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        eval_every=eval_every,
        accumulation_steps=accumulation_steps,
        seed=seed,
        batch_backbone=batch_backbone,
        save_every_epoch=save_every_epoch,
        mlp_lr=mlp_lr,
        noul_lr=noul_lr,
        choice_lr=choice_lr,
        score_lr=score_lr,
        uncertainty_weighting=uncertainty_weighting,
    )

    # Print final eval summary
    last_eval = None
    for entry in reversed(results["history"]):
        if "eval" in entry:
            last_eval = entry["eval"]
            break

    if last_eval:
        print("\n=== Final eval ===", flush=True)
        _print_detailed_eval(last_eval)

    # Save results
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {output}", flush=True)

    return 0


def _print_detailed_eval(eval_stats: dict) -> None:
    """Print formatted detailed eval results."""
    agg = eval_stats["aggregate"]
    print(
        f"  Aggregate: loss={agg['mean_loss']:.4f} acc={agg['accuracy']:.4f} ({agg['n_items']} items)",
        flush=True,
    )

    by_type = eval_stats.get("by_type", {})
    if by_type:
        print("\n  By type:", flush=True)
        for t in sorted(by_type):
            ts = by_type[t]
            print(
                f"    {t:<10} loss={ts['mean_loss']:.4f} acc={ts['accuracy']:.4f} ({ts['n_items']} items)",
                flush=True,
            )

    by_source = eval_stats.get("by_source", {})
    if by_source:
        print(
            f"\n  {'Source':<20} {'Type':<8} {'Loss':>8} {'Acc':>8} {'N':>6}",
            flush=True,
        )
        print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}", flush=True)
        for src in sorted(by_source):
            ss = by_source[src]
            print(
                f"  {src:<20} {ss['type']:<8} {ss['mean_loss']:>8.4f} "
                f"{ss['accuracy']:>8.4f} {ss['n_items']:>6}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
