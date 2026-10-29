#!/usr/bin/env python3
"""Measure inference latency and throughput per option count.

The goal sets two efficiency targets that no experiment currently measures:
at most 100 ms per item at 256 options on one A100, and a 0.6B variant within
3 pp of the 4B variant. Accuracy alone cannot tell whether a readout is
affordable in the many-candidate regime, where the cost model changes shape
(one forward per option, or one per chunk, depending on the readout).

This script times a single forward pass per option count, with the model in
eval mode and inference weights, and reports median/p95 latency plus peak GPU
memory. Option counts span the range the decision APIs advertise: a binary
noul question, a small choice set, the 20-option range where decision models
are documented to degrade, and 256 as the published ceiling.

Usage:
    python model/scripts/bench_latency.py \
        --config model/configs/r2_reranker4b.yaml \
        --model Qwen/Qwen3-Reranker-8B \
        --lora-layers -1 --readout weighted \
        --load-heads model/checkpoints/arch_8b_loraAll_weighted/best_heads.pt \
        --n-options 2 5 20 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.checkpoint_readout import infer_readout
from model.evaluation.latency import DEFAULT_OPTION_COUNTS, make_options, summarise
from model.src.backbone import load_causal_lm, load_encoder, load_qwen35_base
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info

# A deterministic passage of roughly 200 tokens, so latency is driven by the
# option count rather than by how long the context happens to be.
_STATE_WORDS = (
    "The release is scheduled for Friday and the build has two failing tests. "
    "The on-call engineer reports that one failure is a flaky timeout and the "
    "other is a real regression in the payment path. The product owner asks "
    "whether to ship with the known failure, delay the release, or ship behind "
    "a feature flag. "
)
STATE = (_STATE_WORDS * 3).strip()
INSTRUCTIONS = "Which action should the team take?"


def build_model(args: argparse.Namespace, cfg: dict, device: str):
    """Construct the DecisionModel exactly as the training script would."""
    model_name = args.model or cfg.get("model", "Qwen/Qwen3-0.6B")
    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=args.max_length)
    # Fall back to the config's own value.  Passing ``None`` leaves
    # ``DecisionModel`` to use ``gpu_config.max_length``, which ignores the
    # config the command names -- so a run without --max-length would silently
    # benchmark a different length than the config declares.
    if args.max_length:
        max_length = args.max_length
    else:
        max_length = cfg.get("max_length")

    print(f"\nGPU optimization config (model ~{estimate_model_billions(model_name):.1f}B params):")
    print_gpu_info(gpu_config)

    flash_att = gpu_config.flash_attention
    if cfg.get("qwen35_base", False):
        backbone, tokenizer = load_qwen35_base(
            model_name, device=device, freeze=True, flash_attention=flash_att
        )
    elif cfg.get("encoder", False):
        backbone, tokenizer = load_encoder(
            model_name, device=device, freeze=True, flash_attention=flash_att
        )
    else:
        backbone, tokenizer = load_causal_lm(
            model_name, device=device, freeze=True, flash_attention=flash_att
        )

    lora_layers = args.lora_layers if args.lora_layers is not None else cfg.get("lora_layers")
    # The checkpoint decides the readout when neither the CLI nor an explicit
    # --readout says otherwise: a weighted checkpoint benchmarked as `last`
    # times a different network than the one that was trained.
    readout = args.readout or cfg.get("readout", "last")
    if args.readout is None and args.load_heads is not None:
        inferred = infer_readout(
            torch.load(args.load_heads, map_location="cpu", weights_only=True).keys()
        )
        if inferred and inferred != readout:
            print(
                f"readout inferred from the checkpoint: {inferred} "
                f"(config said {readout})",
                flush=True,
            )
            readout = inferred
    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=args.rank if args.rank is not None else cfg.get("rank", 64),
        max_length=max_length,
        gpu_config=gpu_config,
        shared_attention=cfg.get("shared_attention", False),
        readout=readout,
        readout_n=cfg.get("readout_n", 4),
        lora_layers=lora_layers,
        lora_rank=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_targets=tuple(
            (cfg.get("lora_targets", "q_proj,v_proj")).split(",")
            if isinstance(cfg.get("lora_targets", "q_proj,v_proj"), str)
            else cfg.get("lora_targets", ("q_proj", "v_proj"))
        ),
    )
    if args.load_heads:
        model.load_heads(args.load_heads)
        print(f"Loaded heads from {args.load_heads}", flush=True)
    model.eval()
    return model


def time_option_count(
    model: DecisionModel,
    n_options: int,
    repeats: int,
    warmup: int,
    use_cuda: bool,
) -> dict[str, float]:
    """Time one forward pass for a choice question with ``n_options`` options."""
    options = make_options(n_options)
    for _ in range(warmup):
        with torch.no_grad():
            model.forward_choice(STATE, INSTRUCTIONS, options)

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            model.forward_choice(STATE, INSTRUCTIONS, options)
        if use_cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)

    result = summarise(samples)
    result["peak_mem_gb"] = (
        torch.cuda.max_memory_allocated() / 1024**3 if use_cuda else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency and throughput per option count")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-heads", type=Path, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--n-options", type=int, nargs="+", default=list(DEFAULT_OPTION_COUNTS)
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the backbone with reduce-overhead (CUDA graphs)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Write a run record here: device, torch/dtype, effective settings "
            "and every timing. The table is the deliverable; without this the "
            "numbers exist only in the terminal."
        ),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = args.device
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: CUDA unavailable; timings measure CPU inference only.", flush=True)

    model = build_model(args, cfg, device)
    warmup = args.warmup
    if args.compile:
        # The first calls pay compilation and CUDA-graph capture.  Three warmup
        # iterations measure the compiler, not the model.
        warmup = max(warmup, 10)
        # Latency here is launch-bound: the per-call cost is roughly flat from 2
        # to 20 options, which is the arithmetic being a small part of it.
        # CUDA graphs collapse the per-layer kernel launches, which is the only
        # lever that moves the fixed cost rather than the token count.
        model.backbone = torch.compile(
            model.backbone, mode="reduce-overhead", dynamic=False
        )
        print("backbone compiled with reduce-overhead", flush=True)
    print(
        f"readout={getattr(model, 'readout_mode', 'n/a')} "
        f"max_length={model.max_length} options={args.n_options} "
        f"repeats={args.repeats}",
        flush=True,
    )

    record: dict[str, Any] = {
        "model": args.model or cfg.get("model"),
        "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(next(model.backbone.parameters()).dtype),
        "state_tokens": len(model.tokenizer(STATE).input_ids),
        "max_length": model.max_length,
        "readout": getattr(model, "readout_mode", None),
        "compiled": bool(args.compile),
        "warmup": warmup,
        "repeats": args.repeats,
        "timings": {},
    }

    header = f"{'options':>8} {'median_ms':>10} {'p95_ms':>8} {'items/s':>9} {'peak_GB':>8}"
    print("\n" + header)
    print("-" * len(header))
    for n_options in args.n_options:
        stats = time_option_count(model, n_options, args.repeats, warmup, use_cuda)
        print(
            f"{n_options:>8} {stats['median_ms']:>10.1f} {stats['p95_ms']:>8.1f} "
            f"{stats['items_per_s']:>9.1f} {stats['peak_mem_gb']:>8.2f}",
            flush=True,
        )
        record["timings"][str(n_options)] = stats

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2))
        print(f"\nRun record written to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
