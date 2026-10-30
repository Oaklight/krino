#!/usr/bin/env python3
"""Does accuracy depend on state length alone?

Holds the item fixed — same question, same options, same gold label — and varies
only how much neutral text follows the state. `research/13` measures a 30-point
fall from the shortest to the longest quartile of JevBench hard-tier states, but
that is a comparison across items, where length is confounded with everything
else that makes a hard item hard. `scripts/state_length_audit.py` shows why the
direction matters: 25 of 69,287 training choice items reach the tier's median
state length, so a drop here is a drop on inputs the model has never seen, and a
flat line says the tier's long states carry something length alone does not
reproduce.

Usage:
    python model/scripts/state_length_probe.py \
        --config model/configs/r2_reranker4b.yaml \
        --load-heads model/checkpoints/arch_4b_loraAll_32k_s123/epoch_010.pt \
        --lora-layers -1 --readout last \
        --source agnews --items 300 --targets 0,500,1500,4000,8000 \
        --json model/experiments/state_length_agnews_4b.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

import data.pipeline as pipeline_mod  # noqa: E402
import data.sources as sources_mod  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.evaluation.state_padding import pad_state  # noqa: E402
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info  # noqa: E402
from model.src.model_factory import build_decision_model  # noqa: E402


def _score(model: Any, state: str, item: Any, option_texts: list[str], keys: list[str]) -> bool:
    """Whether the argmax over options is the gold key."""
    instructions = item.question.get("instructions", "")
    logits = model.forward_choice(state, instructions, option_texts)[0]
    logits = logits.float().cpu().tolist()
    best = max(range(len(logits)), key=lambda index: logits[index])
    return keys[best] == item.label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--source", default="agnews")
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument(
        "--targets",
        default="0,500,1500,4000,8000",
        help="Total state length in characters; 0 is the untouched state",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--readout-n", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-targets", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--noul-rank", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument(
        "--choice-mode",
        choices=["tower", "cross_encoder"],
        default=None,
        help=(
            "How the choice head scores options (default: the checkpoint's own "
            "shape). The cross-encoder puts the whole state into every option's "
            "prompt, so it reads padding differently from a tower that attends "
            "over the context sequence -- which is the reason to probe both."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_name = args.model or cfg.get("model")
    max_length = int(args.max_length or cfg.get("max_length") or 32768)
    readout = args.readout or cfg.get("readout", "last")
    lora_layers = args.lora_layers if args.lora_layers is not None else cfg.get("lora_layers")
    lora_rank = int(args.lora_rank or cfg.get("lora_rank") or 16)
    lora_alpha = int(args.lora_alpha or cfg.get("lora_alpha") or 32)
    lora_dropout = float(args.lora_dropout if args.lora_dropout is not None else cfg.get("lora_dropout", 0.0))
    lora_targets = tuple(
        part.strip()
        for part in str(args.lora_targets or cfg.get("lora_targets") or "q_proj,v_proj").split(",")
        if part.strip()
    )
    rank = int(args.rank or cfg.get("rank") or 128)
    noul_rank = int(args.noul_rank or cfg.get("noul_rank") or rank)
    mlp_layers = int(args.mlp_layers if args.mlp_layers is not None else cfg.get("mlp_layers", 0))
    choice_mode = args.choice_mode or str(cfg.get("choice_mode") or "tower")

    if args.data_root:
        data_root = args.data_root
        sources_mod.ROOT = data_root
        pipeline_mod.DATA_DIR = data_root / "data" / "benchmarks"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=== State-length probe ===", flush=True)
    print(f"Backbone: {model_name}", flush=True)
    print(f"Heads:    {args.load_heads}", flush=True)

    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=max_length)
    print_gpu_info(gpu_config)

    device = args.device or cfg.get("device")
    model = build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=device,
        use_encoder=bool(cfg.get("encoder", False)),
        rank=rank,
        mlp_layers=mlp_layers,
        noul_rank=noul_rank,
        max_length=max_length,
        shared_attention=bool(cfg.get("shared_attention", True)),
        readout=readout,
        readout_n=int(args.readout_n or cfg.get("readout_n") or 4),
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        choice_mode=choice_mode,
    )
    model.load_heads(args.load_heads)
    model.eval()

    source_cfg = (cfg.get("sources") or {}).get(args.source) or {
        "path": f"data/benchmarks/{args.source}.jsonl"
    }
    _, eval_items = load_source_items(args.source, source_cfg, random.Random(args.seed))
    choice_items = [item for item in eval_items if item.question.get("type") == "choice"]
    rng = random.Random(f"{args.seed}:probe")
    rng.shuffle(choice_items)
    choice_items = choice_items[: args.items]
    print(f"  {len(choice_items)} {args.source} choice items", flush=True)

    targets = [int(part) for part in args.targets.split(",") if part.strip()]
    by_target: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for target in targets:
        correct = 0
        tokens = 0
        for item in choice_items:
            criteria = item.question["criteria"]
            keys = list(criteria.keys())
            texts = [value or key for key, value in criteria.items()]
            state = item.state if target <= 0 else pad_state(item.state, target)
            # The model's own tokeniser, so the length the backbone sees is
            # measured rather than inferred from characters.
            tokens += len(model.tokenizer(f"{state} {item.question.get('instructions', '')}").input_ids)
            if _score(model, state, item, texts, keys):
                correct += 1
        n = len(choice_items)
        by_target[str(target)] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "mean_tokens": tokens / n if n else 0.0,
            "mean_chars": (
                sum(len(item.state) if target <= 0 else len(pad_state(item.state, target)) for item in choice_items) / n
                if n
                else 0.0
            ),
        }
        level = by_target[str(target)]
        print(
            f"  target={target:<6d} acc={level['accuracy']:.4f} "
            f"chars={level['mean_chars']:.0f} tokens={level['mean_tokens']:.0f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    print(f"\nelapsed {elapsed:.1f}s", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.load_heads),
            "config": str(args.config),
            "source": args.source,
            "items": len(choice_items),
            "seed": args.seed,
            "max_length": max_length,
            "readout": readout,
            "choice_mode": choice_mode,
            "lora_layers": lora_layers,
            "elapsed_s": round(elapsed, 1),
            "by_target_chars": by_target,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
