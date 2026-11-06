"""Measure folding a LoRA adapter into the v4 shared-prefix reranker.

The v4 shared-prefix path scores every option of an item off one prefix KV
cache.  A LoRA adapter changes q/v (and, if targeted, the GDN projections), so
the question is whether the adapter can be folded into the base weights at
inference.  If it can, there is no wrapper in the prefix pass, no cache
invalidation, and nothing to optimize.

The script measures, on one GPU and the same JevBench items:

1. folding feasibility: every adapted module is a ``nn.Linear`` and
   ``merge_lora`` removes it;
2. latency per item for the frozen path, the adapter path with wrappers, and
   the merged path;
3. numerical parity between the wrapper path and the merged path: maximum
   absolute difference in the yes/no logit gap, maximum probability difference,
   and the number of items whose predicted label changes.

A nonzero adapter is required for the parity check.  Until #233 produces a
trained v4 adapter, the adapter here is synthetic: ``lora_b`` is filled with a
seeded Gaussian whose scale is reported, and the relative Frobenius norm of
``B A`` against ``W`` is reported so the magnitude is interpretable.

Usage:
    python model/scripts/bench_adapter_fold.py \
        --model Qwen/Qwen3.5-4B \
        --targets q_proj,v_proj \
        --json results/gate/adapter_fold_qv.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.scripts.eval_gate import RawReranker, _raw_options  # noqa: E402
from model.src.lora import LoRALinear, inject_lora, merge_lora  # noqa: E402


def _adapter_stats(model: torch.nn.Module, targets: tuple[str, ...]) -> dict[str, Any]:
    """Adapter count by module suffix/type and the relative delta norm."""
    by_suffix: Counter = Counter()
    by_type: Counter = Counter()
    ratios: list[float] = []
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        by_suffix[name.rsplit(".", 1)[-1]] += 1
        by_type[type(module.base).__name__] += 1
        with torch.no_grad():
            delta = module.lora_b.weight.float() @ module.lora_a.weight.float()
            delta = delta * module.scaling
            base_norm = module.base.weight.float().norm()
            if base_norm > 0:
                ratios.append(float(delta.norm() / base_norm))
    return {
        "count": sum(by_suffix.values()),
        "by_suffix": dict(by_suffix),
        "by_type": dict(by_type),
        "targets": list(targets),
        "relative_delta_norm_mean": statistics.mean(ratios) if ratios else None,
        "relative_delta_norm_max": max(ratios) if ratios else None,
    }


def _fill_adapter(model: torch.nn.Module, scale: float, seed: int) -> None:
    """Deterministic nonzero ``lora_b`` so the adapter changes the output."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                noise = torch.randn(
                    module.lora_b.weight.shape, generator=generator
                ) * scale
                module.lora_b.weight.copy_(noise.to(module.lora_b.weight.dtype))


def _score(
    reranker: RawReranker, items: list[Any]
) -> list[list[float]]:
    results: list[list[float]] = []
    for item in items:
        question = item.question
        options = _raw_options(question)
        results.append(
            reranker.score_many(
                item.state,
                question.get("instructions", ""),
                question["type"],
                options,
            )
        )
    return results


def _time_score(
    reranker: RawReranker,
    items: list[Any],
    warmup: int,
    repeats: int,
    use_cuda: bool,
) -> dict[str, float]:
    for item in items[:warmup]:
        question = item.question
        reranker.score_many(
            item.state,
            question.get("instructions", ""),
            question["type"],
            _raw_options(question),
        )
    samples: list[float] = []
    for _ in range(repeats):
        for item in items:
            question = item.question
            options = _raw_options(question)
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            reranker.score_many(
                item.state,
                question.get("instructions", ""),
                question["type"],
                options,
            )
            if use_cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
    return {
        "items": len(items),
        "repeats": repeats,
        "samples": len(samples),
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "p95_s": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
    }


def _logit(probability: float) -> float:
    clamped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clamped / (1.0 - clamped))


def _parity(
    left: list[list[float]], right: list[list[float]]
) -> dict[str, Any]:
    max_logit = 0.0
    sum_logit = 0.0
    max_prob = 0.0
    count = 0
    flips = 0
    flip_margins: list[float] = []
    for left_item, right_item in zip(left, right):
        if len(left_item) != len(right_item):
            raise SystemExit("option counts differ between conditions")
        for left_score, right_score in zip(left_item, right_item):
            diff = abs(_logit(left_score) - _logit(right_score))
            max_logit = max(max_logit, diff)
            sum_logit += diff
            max_prob = max(max_prob, abs(left_score - right_score))
            count += 1
        left_values = sorted((_logit(score) for score in left_item), reverse=True)
        right_values = sorted((_logit(score) for score in right_item), reverse=True)
        if max(range(len(left_item)), key=left_item.__getitem__) != max(
            range(len(right_item)), key=right_item.__getitem__
        ):
            flips += 1
            # The closer of the two top-1/top-2 margins.  A near-tie flip has
            # a tiny margin on at least one side, so the flip is rounding.
            flip_margins.append(
                min(left_values[0] - left_values[1], right_values[0] - right_values[1])
            )
    return {
        "scored_options": count,
        "items": len(left),
        "max_abs_logit_gap_diff": max_logit,
        "mean_abs_logit_gap_diff": sum_logit / count if count else 0.0,
        "max_abs_prob_diff": max_prob,
        "label_flips": flips,
        "label_flip_fraction": flips / len(left) if left else 0.0,
        "flipped_top2_margin_max": max(flip_margins) if flip_margins else 0.0,
        "flipped_top2_margin_median": (
            statistics.median(flip_margins) if flip_margins else 0.0
        ),
    }


def _fallback_latency(
    reranker: RawReranker,
    items: list[Any],
    repeats: int,
    use_cuda: bool,
) -> dict[str, float]:
    """Per-item cost of the path that re-encodes a full prompt per option."""
    samples: list[float] = []
    for _ in range(repeats):
        for item in items:
            question = item.question
            options = _raw_options(question)
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for label, description in options:
                reranker._score_option(
                    item.state,
                    question.get("instructions", ""),
                    question["type"],
                    label,
                    description,
                )
            if use_cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
    return {
        "items": len(items),
        "repeats": repeats,
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA suffixes; add in_proj_qkv,in_proj_z,in_proj_a,in_proj_b "
        "to cover the GDN projections",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--adapter-scale", type=float, default=0.01)
    parser.add_argument("--adapter-seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fallback-items", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    targets = tuple(part.strip() for part in args.targets.split(",") if part.strip())
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: CUDA unavailable; timings are CPU only", flush=True)

    items = list(load_jevbench())
    if args.limit:
        items = items[: args.limit]
    print(f"model {args.model} items {len(items)} targets {targets}", flush=True)

    reranker = RawReranker(
        args.model,
        args.device,
        loader_type="qwen35_base",
    )
    reranker.shared_prefix = True

    frozen_scores = _score(reranker, items)
    frozen_latency = _time_score(
        reranker, items, args.warmup, args.repeats, use_cuda
    )
    print(f"frozen {frozen_latency['mean_s'] * 1e3:.1f} ms/item", flush=True)

    # ``inject_lora`` initialises ``lora_a`` from the global RNG.  Seed it or
    # every run gets a different adapter, and the parity flips are then a
    # property of the draw rather than of folding.
    torch.manual_seed(args.adapter_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.adapter_seed)
    adapted = inject_lora(
        reranker.model,
        targets=targets,
        rank=args.rank,
        alpha=args.alpha,
    )
    _fill_adapter(reranker.model, args.adapter_scale, args.adapter_seed)
    stats = _adapter_stats(reranker.model, targets)
    print(f"adapted {adapted} modules", flush=True)

    unmerged_scores = _score(reranker, items)
    unmerged_latency = _time_score(
        reranker, items, args.warmup, args.repeats, use_cuda
    )
    print(f"unmerged {unmerged_latency['mean_s'] * 1e3:.1f} ms/item", flush=True)

    start = time.perf_counter()
    merged = merge_lora(reranker.model)
    if use_cuda:
        torch.cuda.synchronize()
    merge_seconds = time.perf_counter() - start
    remaining = sum(
        1 for module in reranker.model.modules() if isinstance(module, LoRALinear)
    )
    print(f"merged {merged} modules in {merge_seconds:.3f} s", flush=True)

    merged_scores = _score(reranker, items)
    merged_latency = _time_score(
        reranker, items, args.warmup, args.repeats, use_cuda
    )
    print(f"merged {merged_latency['mean_s'] * 1e3:.1f} ms/item", flush=True)

    fallback = _fallback_latency(
        reranker, items[: args.fallback_items], 1, use_cuda
    )

    result = {
        "model": args.model,
        "device": "cuda" if use_cuda else "cpu",
        "torch": torch.__version__,
        "items": len(items),
        "adapter": {
            "rank": args.rank,
            "alpha": args.alpha,
            "scale": args.adapter_scale,
            "seed": args.adapter_seed,
            **stats,
        },
        "folding": {
            "expected_modules": adapted,
            "merged_modules": merged,
            "remaining_lora_modules": remaining,
            "foldable": merged == adapted and remaining == 0,
            "merge_seconds": merge_seconds,
            "blockers": [],
        },
        "numerics": {
            "wrapper_vs_merged": _parity(unmerged_scores, merged_scores),
            "frozen_vs_wrapper": _parity(frozen_scores, unmerged_scores),
            "note": (
                "logit gap is log(p/(1-p)) for each option's yes probability, "
                "clamped at 1e-6; a label flip is an argmax change over options"
            ),
        },
        "latency": {
            "frozen": frozen_latency,
            "adapter_wrapper": unmerged_latency,
            "adapter_merged": merged_latency,
            "fallback_per_option": fallback,
        },
    }
    payload = json.dumps(result, indent=1, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
