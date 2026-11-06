"""Train a two-stage pairwise head on top of a frozen backbone.

Stage 1 is the raw backbone's distribution over the full option set; stage 2
duels the shortlist pairwise with the marker head (see ``model.src.pairwise``).
Training pairs the gold with a distractor drawn from the same stage-1
shortlist evaluation uses, so the head sees the distribution of pairs it will
be asked about::

    logits = log(p_base) + alpha_type * pairwise_log_probs

Noul keeps the raw control readout.  The split, fusion, KL/CE weights and seed
handling match the listwise arm.

Usage:
    python model/scripts/train_pairwise.py \
        --items data/benchmarks/residual_train_20000.jsonl \
        --base-probs data/benchmarks/base_probs_20000_qwen35_4b.jsonl \
        --output model/experiments/pairwise_D20K_qwen35 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from data.pipeline import load_jsonl
from model.scripts.train_residual import (
    TYPES,
    _label_index,
    _options,
    parse_lora_targets,
    split_items,
)
from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig
from model.src.pairwise import PairwiseMarkerReadout, stage1_shortlist

TRAINED_TYPES = ("choice", "score")


def _shortlist_alpha(q_type: str) -> bool:
    """Choice pairs average both orders; rubric levels keep their order."""
    return q_type == "choice"


@torch.no_grad()
def evaluate(
    readout: PairwiseMarkerReadout,
    items: list,
    base_probs: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
    m: int,
) -> dict[str, float]:
    """Full-set evaluation; noul keeps the raw base readout."""
    model = readout.model
    model.eval()
    correct = 0.0
    corrected = 0.0
    n = 0
    missed_shortlist = 0
    per_type: dict[str, list[float]] = defaultdict(list)
    for item in items:
        base = base_probs.get(item.id)
        if base is None:
            continue
        target = _label_index(item)
        if target < 0:
            continue
        q_type = item.question["type"]
        if q_type == "noul":
            probs = base
        else:
            options = _options(item.question)
            probs, shortlist = readout.shortlist_probabilities(
                item.state,
                item.question.get("instructions", ""),
                options,
                base,
                float(alphas[q_type]),
                m=m,
                order_sensitive=_shortlist_alpha(q_type),
            )
            if target not in shortlist:
                missed_shortlist += 1
        pred = int(torch.argmax(probs).item())
        k = probs.numel()
        chance = 1.0 / k
        hit = 1.0 if pred == target else 0.0
        correct += hit
        corrected += (hit - chance) / (1.0 - chance)
        per_type[q_type].append((hit - chance) / (1.0 - chance))
        n += 1
    model.train()
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "corrected": corrected / n if n else 0.0,
        "by_type": {t: sum(v) / len(v) for t, v in per_type.items() if v},
        "missed_shortlist": missed_shortlist,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--base-probs", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--option-max-tokens", type=int, default=64)
    parser.add_argument(
        "--shortlist-m",
        type=int,
        default=3,
        help="Stage-1 shortlist size used for training pairs and validation",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-2)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=None,
        help="Inject LoRA into the last N blocks (negative = every block); "
        "omitted leaves the backbone frozen",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-targets",
        type=parse_lora_targets,
        default=("q_proj", "v_proj"),
        help="Comma-separated linear-layer suffixes that receive adapters",
    )
    parser.add_argument(
        "--lora-kl-weight",
        type=float,
        default=0.0,
        help="KL from the frozen base distribution to the pre-fusion readout; "
        "0 leaves the loss unchanged",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the train/val split; keep 42 to match D20K",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    items = load_jsonl(args.items)
    base_records = [
        json.loads(line)
        for line in args.base_probs.read_text().splitlines()
        if line.strip()
    ]
    base_probs = {
        rec["id"]: torch.tensor(
            rec["probs"], dtype=torch.float32, device=args.device
        )
        for rec in base_records
    }
    items = [item for item in items if item.id in base_probs]
    if args.limit:
        items = items[: args.limit]
    train_items, val_items = split_items(items, args.val_fraction, args.split_seed)
    train_items = [
        item for item in train_items if item.question["type"] in TRAINED_TYPES
    ]
    print(
        f"train {len(train_items)} val {len(val_items)} "
        f"types { {t: sum(1 for i in items if i.question['type'] == t) for t in TYPES} }",
        flush=True,
    )

    backbone, tokenizer = load_causal_lm(
        args.model, device=args.device, freeze=True, flash_attention=False
    )
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=True,
        max_length=args.max_length,
        gradient_checkpointing=False,
    )
    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=args.rank,
        mlp_layers=0,
        gpu_config=gpu,
        max_length=args.max_length,
        option_marker=True,
        option_chunk_size=1,
        option_max_tokens=args.option_max_tokens,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
    )
    readout = PairwiseMarkerReadout(model)
    alphas = {t: torch.nn.Parameter(torch.zeros((), device=args.device)) for t in TYPES}
    head_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": list(alphas.values()), "lr": args.alpha_lr},
        ]
    )
    print(f"trainable heads {sum(p.numel() for p in head_params):,}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    best = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_items)
        rng = random.Random(args.seed + epoch)
        rng.shuffle(order)
        total = kl_total = ce_total = 0.0
        started = time.time()
        for index, item in enumerate(order, 1):
            q = item.question
            options = _options(q)
            gold = _label_index(item)
            base = base_probs[item.id]
            if gold < 0:
                continue
            shortlist = stage1_shortlist(base, args.shortlist_m, must_include=gold)
            distractors = [i for i in shortlist if i != gold]
            if not distractors:
                continue
            distractor = rng.choice(distractors)
            loss, kl_value, ce_value = readout.pair_loss(
                item.state,
                q.get("instructions", ""),
                options,
                gold,
                distractor,
                base,
                alphas[q["type"]],
                rng,
                kl_weight=args.kl_weight,
                ce_weight=args.ce_weight,
                order_sensitive=_shortlist_alpha(q["type"]),
                lora_kl_weight=args.lora_kl_weight,
            )
            kl_total += kl_value
            ce_total += ce_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params + list(alphas.values()), 1.0)
            optimizer.step()
            total += float(loss.detach())
            if index % 200 == 0:
                rate = index / (time.time() - started)
                print(
                    f"  epoch {epoch} {index}/{len(order)} loss={total/index:.4f} "
                    f"{rate:.2f} items/s",
                    flush=True,
                )
        metrics = evaluate(
            readout, val_items, base_probs, alphas, args.shortlist_m
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(1, len(order))
        metrics["kl_mean"] = kl_total / max(1, len(order))
        metrics["ce_mean"] = ce_total / max(1, len(order))
        metrics["alpha"] = {t: float(a.detach()) for t, a in alphas.items()}
        history.append(metrics)
        print(f"  val {json.dumps(metrics)}", flush=True)
        if best is None or metrics["corrected"] > best:
            best = metrics["corrected"]
            stale = 0
            best_state = {
                "heads": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                    if "backbone" not in k or k in trainable_names
                },
                "alpha": {t: float(a.detach()) for t, a in alphas.items()},
            }
        else:
            stale += 1
            if stale > args.patience:
                print(f"  early stop at epoch {epoch} (best cc {best:.4f})", flush=True)
                break

    torch.save(
        {
            "heads": best_state["heads"],
            "alpha": best_state["alpha"],
            "config": {
                "backbone": args.model,
                "readout": "pairwise",
                "rank": args.rank,
                "max_length": args.max_length,
                "kl_weight": args.kl_weight,
                "ce_weight": args.ce_weight,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "option_marker": True,
                "option_chunk_size": 1,
                "option_max_tokens": args.option_max_tokens,
                "shortlist_m": args.shortlist_m,
                "lr": args.lr,
                "alpha_lr": args.alpha_lr,
                "lora_layers": args.lora_layers,
                "lora_rank": args.lora_rank if args.lora_layers is not None else None,
                "lora_alpha": args.lora_alpha if args.lora_layers is not None else None,
                "lora_targets": (
                    list(args.lora_targets) if args.lora_layers is not None else None
                ),
                "lora_kl_weight": args.lora_kl_weight,
            },
            "history": history,
        },
        args.output / "pairwise.pt",
    )
    (args.output / "history.json").write_text(json.dumps(history, indent=1))
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"max_memory_allocated {peak:.1f} GiB", flush=True)
    print(f"saved {args.output/'pairwise.pt'} best_cc={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
