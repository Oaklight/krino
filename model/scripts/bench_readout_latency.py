#!/usr/bin/env python3
"""Measure wall-clock and estimated FLOPs for the two option readouts.

The pooled readout encodes the context and the whole option batch in two
backbone forwards.  The option-marker readout encodes the context once and
then one suffix forward per chunk, so it uses ``1 + ceil(K / chunk)``
forwards.  Which one is faster depends on whether the workload is launch-bound
(short sequences, many forwards) or compute-bound (long context, large suffix
attention).  This script measures both instead of assuming.

Usage:
    python model/scripts/bench_readout_latency.py \
        --option-counts 5,32,64,128,256 \
        --chunk-sizes 32,64,128,256 \
        --context-tokens 64,2000,8000

The FLOP numbers are estimates from the transformer shape, not profiler
counters.  They include the linear layers (``2 * params * tokens``), the
prefix self-attention, the option batch or suffix attention, and the readout
head.  Use them to see the scaling; use the wall-clock column for latency.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig, estimate_model_billions


def build_models(
    model_name: str,
    device: str,
    chunk_sizes: list[int],
    rank: int,
) -> tuple[dict[str, DecisionModel], object]:
    gpu = GPUConfig.auto_detect(estimate_model_billions(model_name))
    backbone, tokenizer = load_causal_lm(
        model_name,
        device=device,
        freeze=True,
        flash_attention=gpu.flash_attention,
    )
    models: dict[str, DecisionModel] = {
        "pooled": DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            rank=rank,
            gpu_config=gpu,
            shared_attention=True,
            option_marker=False,
        )
    }
    for chunk in chunk_sizes:
        models[f"marker_b{chunk}"] = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            rank=rank,
            gpu_config=gpu,
            shared_attention=True,
            option_marker=True,
            option_chunk_size=chunk,
        )
    return models, tokenizer


def _time_call(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def estimate_flops(
    model: DecisionModel,
    context_tokens: int,
    option_tokens: int,
    n_options: int,
    chunk_size: int,
    rank: int = 64,
) -> float:
    """Rough transformer FLOP estimate for one forward_choice call."""
    config = model.backbone.config
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    params = sum(p.numel() for p in model.backbone.parameters())

    if model.option_marker:
        # Prefix forward + ceil(K/B) suffix forwards.  The linear work covers
        # the same L + K*S tokens as the pooled batch; the extra term is the
        # suffix's attention to the prefix in every layer.
        linear = 2 * params * (context_tokens + n_options * option_tokens)
        prefix_attn = 2 * n_layers * hidden * context_tokens**2
        suffix_attn = 4 * n_layers * hidden * n_options * option_tokens * context_tokens
        head = 2 * n_options * hidden * (3 * rank)
        return linear + prefix_attn + suffix_attn + head

    # Pooled: one context forward and one padded option batch forward.  The
    # option batch pays no attention to the context; the head cross-attends.
    linear = 2 * params * (context_tokens + n_options * option_tokens)
    prefix_attn = 2 * n_layers * hidden * context_tokens**2
    option_self = 2 * n_layers * hidden * n_options * option_tokens**2
    head = 4 * n_options * context_tokens * rank  # QK^T + AV at rank r
    return linear + prefix_attn + option_self + head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--option-counts", default="5,32,64,128,256")
    parser.add_argument("--chunk-sizes", default="32,64,128,256")
    parser.add_argument("--context-tokens", default="64,2000,8000")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    option_counts = [int(v) for v in args.option_counts.split(",") if v.strip()]
    chunk_sizes = [int(v) for v in args.chunk_sizes.split(",") if v.strip()]
    context_targets = [int(v) for v in args.context_tokens.split(",") if v.strip()]

    models, tokenizer = build_models(
        args.model, args.device, chunk_sizes, args.rank
    )
    for model in models.values():
        model.eval()

    instructions = "Which candidate answers the question?"
    print(
        f"{'ctx':>6} {'K':>4} {'readout':>12} {'fwd':>4} "
        f"{'ms':>8} {'TFLOPs':>8}",
        flush=True,
    )
    for context_target in context_targets:
        # One token per repeated word, so the target is close to the actual
        # token count; print the measured count.
        state = "context " * max(1, context_target - 2)
        context_tokens = len(
            tokenizer(f"{state} {instructions}", add_special_tokens=False)["input_ids"]
        )
        for k in option_counts:
            options = [f"candidate {i}: description {i}" for i in range(k)]
            option_tokens = max(
                len(tokenizer(opt, add_special_tokens=False)["input_ids"])
                for opt in options
            )
            for name, model in models.items():
                chunk = model.option_chunk_size if model.option_marker else k
                forwards = 2 if not model.option_marker else 1 + (k + chunk - 1) // chunk
                elapsed = _time_call(
                    lambda: model.forward_choice(state, instructions, options),
                    args.warmup,
                    args.repeats,
                )
                flops = estimate_flops(
                    model, context_tokens, option_tokens, k, chunk, args.rank
                )
                print(
                    f"{context_tokens:>6} {k:>4} {name:>12} {forwards:>4} "
                    f"{elapsed * 1000:>8.1f} {flops / 1e12:>8.2f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
