"""Train a listwise set head on top of a frozen backbone.

The head scores a whole option set in one sequence (see
``model.src.listwise``), and is fused with the base distribution exactly like
the per-option marker arm::

    logits = log(p_base) + alpha_type * listwise_log_probs

so the listwise and two-stage arms differ only in how the head reads the
options, not in the fusion, the data or the split.  The head contributes
log-probabilities, the same scale evaluation uses after averaging shifts.

Training samples a shuffled, gold-containing subset of size 2..K per item, so
the same head can rank a stage-1 shortlist.  At evaluation, choice items
average four cyclic shifts of the option order; score items keep the natural
order because rubric levels are ordered.

Usage:
    python model/scripts/train_listwise.py \
        --items data/benchmarks/residual_train_20000_jev.jsonl \
        --base-probs data/benchmarks/base_probs_20000_qwen35.jsonl \
        --output model/experiments/listwise_b1 --epochs 4
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
import torch.nn.functional as F

from data.pipeline import load_jsonl
from model.scripts.train_residual import (
    TYPES,
    _label_index,
    _options,
    parse_lora_targets,
    split_items,
    soft_target,
)
from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig
from model.src.listwise import ListwiseMarkerReadout

TRAINED_TYPES = ("choice", "score")


def _subset_probs(base: torch.Tensor, subset: list[int]) -> torch.Tensor:
    """Base distribution restricted to a subset and renormalised."""
    picked = base[torch.tensor(subset, device=base.device)].clamp_min(1e-6)
    return picked / picked.sum()


def fused_logits(
    base: torch.Tensor, head: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """``log(p_base) + alpha * head`` over the same subset."""
    return torch.log(base.clamp_min(1e-6)) + alpha * head


def item_loss(
    readout: ListwiseMarkerReadout,
    item,
    base: torch.Tensor,
    alpha: torch.Tensor,
    rng: random.Random,
    kl_weight: float,
    ce_weight: float,
    min_subset: int,
    shuffle_score: bool,
    max_subset: int | None = None,
    lora_kl_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """Listwise loss on one item over a sampled subset.

    ``lora_kl_weight`` anchors the readout distribution (before fusion) to the
    frozen base distribution.  The fused KL above already constrains the
    output; this term constrains the backbone readout the adapter changes, and
    it stays off by default so a frozen run is unchanged.
    """
    q = item.question
    options = _options(q)
    gold = _label_index(item)
    if gold < 0:
        zero = torch.zeros((), device=base.device)
        return zero, 0.0, 0.0
    if q["type"] == "choice":
        subset = ListwiseMarkerReadout.sample_subset(
            len(options), gold, rng, min_size=min_subset, max_size=max_subset
        )
    elif shuffle_score:
        subset = ListwiseMarkerReadout.sample_subset(
            len(options), gold, rng, min_size=min_subset, max_size=max_subset
        )
    else:
        # Rubric levels are ordered; shuffling them breaks the scale.
        subset = list(range(len(options)))
    # The head contributes log-probabilities, the same scale evaluation uses
    # after averaging shifts, so one alpha means the same thing in both.
    head = torch.log_softmax(
        readout.score(
            item.state,
            q.get("instructions", ""),
            [options[index] for index in subset],
        ).reshape(-1),
        dim=-1,
    )
    subset_base = _subset_probs(base, subset).to(head.device)
    logits = fused_logits(subset_base, head, alpha)
    log_p = F.log_softmax(logits, dim=-1)
    kl = torch.sum(subset_base * (torch.log(subset_base) - log_p))
    target = subset.index(gold)
    teacher = soft_target(item, len(options))
    if teacher is not None:
        teacher = teacher[torch.tensor(subset)].to(logits.device)
        teacher = teacher / teacher.sum().clamp_min(1e-6)
        ce = torch.sum(teacher * (torch.log(teacher.clamp_min(1e-6)) - log_p))
    else:
        ce = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target], device=logits.device),
        )
    loss = kl_weight * kl + ce_weight * ce
    if lora_kl_weight:
        # ``head`` is log q over the same subset, so this is KL(base || q).
        anchor = torch.sum(subset_base * (torch.log(subset_base) - head))
        loss = loss + lora_kl_weight * anchor
    return loss, float(kl.detach()), float(ce.detach())


@torch.no_grad()
def evaluate(
    readout: ListwiseMarkerReadout,
    items: list,
    base_probs: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
    shifts: int,
    max_set: int = 20,
) -> dict[str, float]:
    """Full-set evaluation.

    Noul keeps the raw base readout.  Choice averages ``shifts`` cyclic
    shifts of the option order; score keeps the natural order, because rubric
    levels are ordered and a shift would reorder the scale.
    """
    model = readout.model
    model.eval()
    correct = 0.0
    corrected = 0.0
    n = 0
    capped = 0
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
            scored_options = options
            if not readout.fits(options):
                # One sequence cannot hold the whole set; score a gold-anchored
                # subset so the item still counts, and report how many did.
                capped += 1
                others = [i for i in range(len(options)) if i != target]
                keep = sorted([target] + others[: max_set - 1])
                scored_options = [options[i] for i in keep]
                base = _subset_probs(base, keep)
                target = keep.index(target)
            item_shifts = shifts if q_type == "choice" else 1
            probs = readout.score_shifted(
                item.state,
                item.question.get("instructions", ""),
                scored_options,
                shifts=item_shifts,
            ).reshape(-1)
            probs = torch.softmax(
                fused_logits(base.to(probs.device), torch.log(probs), alphas[q_type]),
                dim=-1,
            )
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
        "capped_sets": capped,
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
    parser.add_argument(
    "--min-subset",
    type=int,
    default=2,
    help="Smallest gold-containing subset sampled during training",
    )
    parser.add_argument(
    "--max-subset",
    type=int,
    default=20,
    help="Largest option subset one sequence scores; also the eval "
    "fallback for sets that do not fit max_length",
    )
    parser.add_argument(
    "--shuffle-score",
    action="store_true",
    help="Also shuffle score items; off because rubric levels are ordered",
    )
    parser.add_argument(
    "--eval-shifts",
    type=int,
    default=4,
    help="Cyclic shifts averaged at evaluation",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
    "--split-seed",
    type=int,
    default=42,
    help="Seed for the train/val split.  Keep it at 42 to match D20K "
    "when --seed varies the training run, or the seeds stop being "
    "replications of one recipe.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    items = load_jsonl(args.items)
    base_records = [
        json.loads(line) for line in args.base_probs.read_text().splitlines() if line.strip()
    ]
    base_probs = {
        rec["id"]: torch.tensor(rec["probs"], dtype=torch.float32) for rec in base_records
    }
    items = [item for item in items if item.id in base_probs]
    if args.limit:
        items = items[: args.limit]
    # The same call as train_residual, so the val split matches D20K.  The
    # split seed is separate from the training seed: a replication seed must
    # vary init and item order, not which items are held out.
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
    readout = ListwiseMarkerReadout(model)
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
            loss, kl_value, ce_value = item_loss(
                readout,
                item,
                base_probs[item.id],
                alphas[item.question["type"]],
                rng,
                args.kl_weight,
                args.ce_weight,
                args.min_subset,
                args.shuffle_score,
                args.max_subset,
                args.lora_kl_weight,
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
            readout,
            val_items,
            base_probs,
            alphas,
            args.eval_shifts,
            args.max_subset,
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
                "readout": "listwise",
                "rank": args.rank,
                "max_length": args.max_length,
                "kl_weight": args.kl_weight,
                "ce_weight": args.ce_weight,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "option_marker": True,
                "option_chunk_size": 1,
                "option_max_tokens": args.option_max_tokens,
                "min_subset": args.min_subset,
                "max_subset": args.max_subset,
                "shuffle_score": args.shuffle_score,
                "eval_shifts": args.eval_shifts,
                "eval_shifts_types": ["choice"],
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
        args.output / "listwise.pt",
    )
    (args.output / "history.json").write_text(json.dumps(history, indent=1))
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"max_memory_allocated {peak:.1f} GiB", flush=True)
    print(f"saved {args.output/'listwise.pt'} best_cc={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
