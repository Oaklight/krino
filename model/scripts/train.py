#!/usr/bin/env python3
"""Train decision heads on frozen backbone."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import load_causal_lm, load_encoder
from model.src.decision_model import DecisionModel
from model.data.pipeline import load_jsonl
from model.training.supervised import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train decision heads")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Backbone model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-data", nargs="+", required=True, help="Training JSONL files")
    parser.add_argument("--eval-data", nargs="+", required=True, help="Evaluation JSONL files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=64, help="Attention head rank")
    parser.add_argument("--rival-aware", action="store_true", help="Use rival-aware attention")
    parser.add_argument("--encoder", action="store_true", help="Use encoder backbone (ModernBERT) instead of causal LM")
    parser.add_argument("--max-train", type=int, default=None, help="Max training items")
    parser.add_argument("--max-eval", type=int, default=None, help="Max eval items")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading backbone: {args.model} ({'encoder' if args.encoder else 'causal'})")
    if args.encoder:
        backbone, tokenizer = load_encoder(args.model, device=args.device, freeze=True)
    else:
        backbone, tokenizer = load_causal_lm(args.model, device=args.device, freeze=True)

    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=args.rank,
        rival_aware=args.rival_aware,
    )
    print(f"Trainable: {model.trainable_parameters():,} params")
    print(f"Frozen:    {model.frozen_parameters():,} params")

    train_items = []
    for path in args.train_data:
        items = load_jsonl(Path(path))
        train_items.extend(items)
    if args.max_train:
        random.shuffle(train_items)
        train_items = train_items[:args.max_train]

    eval_items = []
    for path in args.eval_data:
        items = load_jsonl(Path(path))
        eval_items.extend(items)
    if args.max_eval:
        eval_items = eval_items[:args.max_eval]

    print(f"Train: {len(train_items)} items, Eval: {len(eval_items)} items")

    # Filter to supported types
    supported = {"noul", "choice", "score"}
    train_items = [it for it in train_items if it.question.get("type") in supported]
    eval_items = [it for it in eval_items if it.question.get("type") in supported]
    print(f"After filtering: Train {len(train_items)}, Eval {len(eval_items)}")

    results = train(
        model=model,
        train_items=train_items,
        eval_items=eval_items,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Final eval on all benchmarks
    print(f"\nFinal eval accuracy: {results['history'][-1].get('eval', {}).get('accuracy', 'N/A')}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
