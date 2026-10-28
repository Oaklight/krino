#!/usr/bin/env python3
"""Evaluate a trained Krino checkpoint as the option set grows.

Banking77 has 77 intents, so a decision question starts with up to 77
candidates.  This script builds harder versions of the same eval questions by
keeping the gold intent and padding the option set with other intents, then
measures top-1 accuracy at each requested option count.  It runs the same
option sets through either readout:

    # pooled option embeddings + AttentionHead
    python model/scripts/eval_option_count.py \\
        --config model/configs/r2_reranker0.6b.yaml \\
        --load-heads model/checkpoints/sweep_reranker0.6b_r64_mlp0/epoch_005.pt \\
        --output model/experiments/optcount_pooled.json

    # option-marker readout
    python model/scripts/eval_option_count.py \\
        --config model/configs/r2_reranker0.6b.yaml --option-marker \\
        --load-heads model/experiments/optmarker_frozen/epoch_005.pt \\
        --output model/experiments/optcount_marker.json

Above 77 options the distractor pool runs out of real Banking77 intents.  The
script then adds numbered variants of the existing intents, each with a
distinguishing description, so the count can reach 256 without duplicating the
gold option.  Those variants are a stress test, not a claim about any real
taxonomy; they are reported alongside the 4-77 range where every distractor is
a real intent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jsonl
from data.sources import load_source_items, resolve_path
from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig, estimate_model_billions


def build_option_pool(items: list) -> dict[str, str]:
    """Collect every Banking77 intent key and description."""
    pool: dict[str, str] = {}
    for item in items:
        criteria = item.question.get("criteria")
        if item.question.get("type") != "choice" or not isinstance(criteria, dict):
            continue
        for key, description in criteria.items():
            pool[key] = description or key
    return pool


def extend_pool(pool: dict[str, str], target: int) -> dict[str, str]:
    """Add numbered variants until the pool can supply ``target`` options.

    The variants keep the original description and append a queue qualifier.
    They are only used above the real Banking77 class count.
    """
    extended = dict(pool)
    bases = list(pool.items())
    variant = 0
    while len(extended) < target:
        key, description = bases[variant % len(bases)]
        round_no = variant // len(bases) + 1
        new_key = f"{key}__alt{round_no}"
        extended[new_key] = (
            f"{description} (alternate support queue {round_no}; "
            "same subject area, separate handling team)"
        )
        variant += 1
    return extended


def base_key(key: str) -> str:
    return key.split("__alt", 1)[0]


def sample_options(
    gold_key: str,
    gold_text: str,
    pool: dict[str, str],
    k: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Return ``k`` (key, text) options including the gold one."""
    candidates = [
        (key, text)
        for key, text in pool.items()
        if key != gold_key and base_key(key) != base_key(gold_key)
    ]
    if len(candidates) < k - 1:
        raise ValueError(
            f"pool has {len(candidates)} distractors, need {k - 1}"
        )
    chosen = rng.sample(candidates, k - 1)
    options = [(gold_key, gold_text), *chosen]
    rng.shuffle(options)
    return options


def infer_checkpoint_arch(path: Path) -> dict[str, Any]:
    """Read head-shape facts from a checkpoint's keys.

    A checkpoint does not carry its training config, and ``load_state_dict``
    is lenient, so choosing the wrong head shape can leave part of the head
    randomly initialised without an error.  The key names and shapes are
    enough to recover the three settings that change the head layout.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    arch: dict[str, Any] = {}
    choice_q = state.get("choice_head.attention.query.weight")
    score_q = state.get("score_head.attention.query.weight")
    if choice_q is not None and score_q is not None:
        arch["shared_attention"] = torch.equal(choice_q, score_q)
    if "noul_head.proj.1.weight" in state:
        arch["noul_rank"] = int(state["noul_head.proj.1.weight"].shape[0])
    if any(key.startswith("marker_head.") for key in state):
        arch["option_marker"] = True
    return arch


def build_model(args: argparse.Namespace, cfg: dict[str, Any]) -> DecisionModel:
    model_name = args.model or cfg.get("model", "Qwen/Qwen3-Reranker-0.6B")
    shared_attention = cfg.get("shared_attention", False)
    noul_rank = cfg.get("noul_rank")
    option_marker = args.option_marker
    if args.load_heads:
        arch = infer_checkpoint_arch(args.load_heads)
        shared_attention = arch.get("shared_attention", shared_attention)
        noul_rank = arch.get("noul_rank", noul_rank)
        option_marker = arch.get("option_marker", option_marker)
        print(f"Checkpoint architecture: {arch}", flush=True)

    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    backbone, tokenizer = load_causal_lm(
        model_name,
        device=args.device,
        freeze=True,
        flash_attention=gpu_config.flash_attention,
    )
    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=args.rank if args.rank is not None else cfg.get("rank", 64),
        rival_aware=cfg.get("rival_aware", False),
        mlp_layers=cfg.get("mlp_layers", 0),
        mlp_dim=cfg.get("mlp_dim"),
        noul_rank=noul_rank,
        gpu_config=gpu_config,
        shared_attention=shared_attention,
        readout=cfg.get("readout", "last"),
        readout_n=cfg.get("readout_n", 4),
        option_marker=option_marker,
        option_chunk_size=args.option_chunk_size,
        option_max_tokens=args.option_max_tokens,
    )
    if args.load_heads:
        print(f"Loading heads from {args.load_heads}", flush=True)
        model.load_heads(args.load_heads)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-heads", type=Path, default=None)
    parser.add_argument("--option-marker", action="store_true")
    parser.add_argument("--option-chunk-size", type=int, default=32)
    parser.add_argument("--option-max-tokens", type=int, default=64)
    parser.add_argument(
        "--option-counts",
        default="4,8,16,32,64,77,128,256",
        help="Comma-separated option counts, ascending",
    )
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    counts = sorted({int(part) for part in args.option_counts.split(",") if part.strip()})
    if counts[0] < 2:
        parser.error("option counts must be at least 2")

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    rng = random.Random(args.seed)
    train_items, eval_items = load_source_items(
        "banking77", cfg["sources"]["banking77"], rng
    )
    eval_items = [
        item
        for item in eval_items
        if item.question.get("type") == "choice"
        and item.label in item.question.get("criteria", {})
    ][: args.max_items]
    if not eval_items:
        print("No Banking77 eval items found", file=sys.stderr)
        return 1

    all_items = load_jsonl(resolve_path(cfg["sources"]["banking77"]["path"]))
    real_pool = build_option_pool(all_items)
    # Keep a few spare variants so excluding the gold and its own variants
    # still leaves K-1 distractors at the largest requested count.
    pool = extend_pool(real_pool, max(counts) + 8)
    print(
        f"Banking77: {len(eval_items)} eval items, {len(real_pool)} real intents, "
        f"{len(pool)}-option stress pool",
        flush=True,
    )

    model = build_model(args, cfg)
    readout = "option-marker" if args.option_marker else "pooled-attention"
    print(f"Readout: {readout}", flush=True)

    by_k: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    with torch.no_grad():
        for k in counts:
            correct = 0
            n = 0
            for index, item in enumerate(eval_items):
                criteria = item.question["criteria"]
                gold_key = item.label
                gold_text = criteria.get(gold_key) or gold_key
                item_rng = random.Random(f"{args.seed}:{item.id}:{k}")
                active_pool = real_pool if k <= len(real_pool) else pool
                options = sample_options(
                    gold_key, gold_text, active_pool, k, item_rng
                )
                logits = model.forward_choice(
                    item.state,
                    item.question.get("instructions", ""),
                    [text for _, text in options],
                )
                pred = int(logits.argmax(dim=-1).item())
                correct += options[pred][0] == gold_key
                n += 1
            by_k[str(k)] = {
                "accuracy": correct / max(n, 1),
                "correct": correct,
                "n": n,
            }
            print(
                f"  K={k:>3}: acc={correct / max(n, 1):.4f} "
                f"({correct}/{n})",
                flush=True,
            )

    result = {
        "readout": readout,
        "checkpoint": str(args.load_heads) if args.load_heads else None,
        "config": str(args.config),
        "option_chunk_size": args.option_chunk_size,
        "n_items": len(eval_items),
        "elapsed_s": round(time.monotonic() - started, 1),
        "by_option_count": by_k,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Results saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
