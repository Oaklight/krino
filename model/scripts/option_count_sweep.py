#!/usr/bin/env python3
"""Measure accuracy against option count on a fixed item set.

Goal item 3 asks for accuracy at >20 options to sit within 5 pp of accuracy at
<=5 options. Comparing the natural buckets cannot answer that, because on this
corpus the >20 bucket is one benchmark (banking77, 80 options) and the <=5 bucket
is thirteen others, so the difference measures the tasks rather than the count.

This script holds the items fixed and varies only K: the same N items, each
padded to every requested K with distractors drawn from the source's own option
inventory, scored once per K. Padding keeps the gold key and every original
option verbatim, so the label stays valid without new annotation — it uses
:func:`data.k_augment.pad_choice_item`, the same function training uses.

The distractor pool comes from the source file's option texts (train and eval),
never from the item being scored, and the item's own options are always excluded,
so a padded item cannot contain its gold text twice.

Usage:
    python model/scripts/option_count_sweep.py \
        --config model/configs/r2_reranker4b.yaml \
        --load-heads model/checkpoints/arch_4b_loraAll_32k_s123/epoch_010.pt \
        --source banking77 --items 300 --ks 4,8,16,32,64,80 \
        --lora-layers -1 --readout last \
        --json model/experiments/optcount_4b_ep10.json
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
from data.k_augment import pad_choice_item, shrink_choice_item  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info  # noqa: E402
from model.src.model_factory import build_decision_model  # noqa: E402


def _option_pool(path: Path) -> list[str]:
    """Every distinct option text in a source file, for use as distractors."""
    texts: dict[str, None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            question = (json.loads(line).get("question") or {})
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                for value in criteria.values():
                    if value:
                        texts.setdefault(str(value), None)
    return list(texts)


def _score(model: Any, item: Any, option_texts: list[str], keys: list[str]) -> bool:
    """Whether the argmax over options is the gold key."""
    state = item.state
    instructions = item.question.get("instructions", "")
    logits = model.forward_choice(state, instructions, option_texts)[0]
    logits = logits.float().cpu().tolist()
    best = max(range(len(logits)), key=lambda index: logits[index])
    return keys[best] == item.label


def _at_k(item: Any, target_k: int, pool: dict[str, list[str]], rng: random.Random) -> Any:
    """The item rendered with exactly ``target_k`` options where the corpus allows."""
    shrunk = shrink_choice_item(item, target_k, rng)
    return pad_choice_item(shrunk, target_k, pool, rng)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--source", default="banking77")
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--ks", default="4,8,16,32,64,80")
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

    if args.data_root:
        data_root = args.data_root
        sources_mod.ROOT = data_root
        pipeline_mod.DATA_DIR = data_root / "data" / "benchmarks"
    data_dir = pipeline_mod.DATA_DIR

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=== Option-count sweep ===", flush=True)
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
    )
    model.load_heads(args.load_heads)
    model.eval()

    source_cfg = (cfg.get("sources") or {}).get(args.source) or {"path": f"data/benchmarks/{args.source}.jsonl"}
    _, eval_items = load_source_items(args.source, source_cfg, random.Random(args.seed))
    choice_items = [item for item in eval_items if item.question.get("type") == "choice"]
    rng = random.Random(f"{args.seed}:sweep")
    rng.shuffle(choice_items)
    choice_items = choice_items[: args.items]
    print(f"  {len(choice_items)} {args.source} choice items", flush=True)

    pool_texts = _option_pool(data_dir / f"{args.source}.jsonl")
    pool = {args.source: pool_texts}
    print(f"  distractor pool: {len(pool_texts)} texts", flush=True)

    ks = [int(part) for part in args.ks.split(",") if part.strip()]
    by_k: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for k in ks:
        correct = 0
        changed = 0
        for item in choice_items:
            probe = _at_k(item, k, pool, random.Random(f"{args.seed}:{k}:{item.id}"))
            if probe is not item or len(probe.question["criteria"]) != len(item.question["criteria"]):
                changed += 1
            criteria = probe.question["criteria"]
            keys = list(criteria.keys())
            texts = [value or key for key, value in criteria.items()]
            if _score(model, probe, texts, keys):
                correct += 1
        n = len(choice_items)
        by_k[str(k)] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "changed": changed,
            "mean_options": sum(len(item.question["criteria"]) for item in choice_items) / n if n else 0.0,
            "chance": 1.0 / k,
        }
        print(f"  K={k:<4d} acc={by_k[str(k)]['accuracy']:.4f} changed={changed}/{n}", flush=True)

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
            "lora_layers": lora_layers,
            "distractor_pool_size": len(pool_texts),
            "elapsed_s": round(elapsed, 1),
            "by_option_count": by_k,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
