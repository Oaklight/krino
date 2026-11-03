"""Train a zero-gated residual head on top of the base reranker.

`logits = log(p_base) + alpha_type * head_logits`, with every `alpha_type`
initialised to 0, so step 0 reproduces the base distribution exactly.  The
loss is a KL anchor to `p_base` plus a small CE term; if CE dominates, the
residual relearns the task and loses the base.

Base probabilities come from `precompute_base_probs.py` so training never runs
the 4B scorer.

Usage:
    python model/scripts/train_residual.py \
        --items data/benchmarks/residual_train_5000.jsonl \
        --base-probs data/benchmarks/base_probs_5000.jsonl \
        --output model/experiments/residual_b1 \
        --epochs 4 --patience 1
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
from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig

TYPES = ("noul", "choice", "score")


def _options(question: dict) -> list[str]:
    q_type = question["type"]
    if q_type == "noul":
        return ["no", "yes"]
    criteria = question["criteria"]
    if q_type == "choice":
        return [value or key for key, value in criteria.items()]
    return [str(level) for level in criteria]


def _label_index(item) -> int:
    q = item.question
    if q["type"] == "noul":
        return 1 if bool(item.label) else 0
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return keys.index(item.label) if item.label in keys else -1
    return int(round(float(item.label)))


def _stratum(item) -> tuple[str, int]:
    n = len(_options(item.question))
    return (item.question["type"], min(n, 8))


def split_items(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """Stratified split by (type, K bucket)."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    for item in items:
        buckets[_stratum(item)].append(item)
    train, val = [], []
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def head_logits(model: DecisionModel, item) -> torch.Tensor:
    q = item.question
    instructions = q.get("instructions", "")
    if q["type"] == "noul":
        logit = model.forward_noul(item.state, instructions).reshape(1)
        return torch.stack([torch.zeros_like(logit), logit], dim=-1)
    if q["type"] == "choice":
        return model.forward_choice(item.state, instructions, _options(q))
    return model.forward_score(item.state, instructions, _options(q))


def soft_target(item, n_options: int) -> torch.Tensor | None:
    """Teacher distribution over options, when the item carries one.

    Returns ``None`` when the item has no teacher_probs or when the labels do
    not line up with the option keys, so the caller falls back to a hard label.
    """
    probs = getattr(item, "teacher_probs", None)
    if not probs or len(probs) != n_options:
        return None
    q = item.question
    if q["type"] == "choice":
        keys = [str(key) for key in q["criteria"]]
    elif q["type"] == "score":
        keys = [str(i) for i in range(len(q["criteria"]))]
    else:
        keys = ["false", "true"]
    if any(key not in probs for key in keys):
        return None
    values = torch.tensor([float(probs[key]) for key in keys], dtype=torch.float32)
    total = float(values.sum())
    if total <= 0:
        return None
    return values / total


def fusion_logits(base: torch.Tensor, head: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.log(base.clamp_min(1e-6)) + alpha * head


def item_loss(
    model: DecisionModel,
    item,
    base_probs: torch.Tensor,
    alphas: dict[str, torch.Tensor],
    kl_weight: float,
    ce_weight: float,
) -> tuple[torch.Tensor, float, float]:
    q_type = item.question["type"]
    head = head_logits(model, item).reshape(-1)
    base = base_probs.to(head.device)
    logits = fusion_logits(base, head, alphas[q_type])
    log_p = F.log_softmax(logits, dim=-1)
    kl = torch.sum(base * (torch.log(base.clamp_min(1e-6)) - log_p))
    target = _label_index(item)
    soft = soft_target(item, logits.numel())
    if soft is not None:
        # Teacher targets keep the label domain; the hard label only reports.
        ce = torch.sum(
            soft.to(logits.device) * (torch.log(soft.clamp_min(1e-6).to(logits.device)) - log_p)
        )
    elif target >= 0:
        ce = F.cross_entropy(
            logits.unsqueeze(0), torch.tensor([target], device=logits.device)
        )
    else:
        ce = torch.zeros((), device=logits.device)
    return kl_weight * kl + ce_weight * ce, float(kl.detach()), float(ce.detach())


@torch.no_grad()
def evaluate(
    model: DecisionModel,
    items: list,
    base_probs: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    correct = 0.0
    corrected = 0.0
    n = 0
    per_type: dict[str, list[float]] = defaultdict(list)
    for item in items:
        base = base_probs.get(item.id)
        if base is None:
            continue
        head = head_logits(model, item).reshape(-1)
        logits = fusion_logits(base.to(head.device), head, alphas[item.question["type"]])
        pred = int(torch.argmax(logits).item())
        target = _label_index(item)
        if target < 0:
            continue
        k = logits.numel()
        chance = 1.0 / k
        hit = 1.0 if pred == target else 0.0
        correct += hit
        corrected += (hit - chance) / (1.0 - chance)
        per_type[item.question["type"]].append((hit - chance) / (1.0 - chance))
        n += 1
    model.train()
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "corrected": corrected / n if n else 0.0,
        "by_type": {t: sum(v) / len(v) for t, v in per_type.items() if v},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--base-probs", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-2)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_jsonl(args.items)
    base_records = [json.loads(line) for line in args.base_probs.read_text().splitlines() if line.strip()]
    base_probs = {
        rec["id"]: torch.tensor(rec["probs"], dtype=torch.float32) for rec in base_records
    }
    items = [item for item in items if item.id in base_probs]
    if args.limit:
        items = items[: args.limit]
    train_items, val_items = split_items(items, args.val_fraction, args.seed)
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
    )
    alphas = {t: torch.nn.Parameter(torch.zeros((), device=args.device)) for t in TYPES}
    head_params = [p for p in model.parameters() if p.requires_grad]
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
        random.Random(args.seed + epoch).shuffle(order)
        total = 0.0
        kl_total = 0.0
        ce_total = 0.0
        started = time.time()
        for index, item in enumerate(order, 1):
            loss, kl_value, ce_value = item_loss(
                model, item, base_probs[item.id], alphas, args.kl_weight, args.ce_weight
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
        metrics = evaluate(model, val_items, base_probs, alphas)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(1, len(order))
        metrics["kl_mean"] = kl_total / max(1, len(order))
        metrics["ce_mean"] = ce_total / max(1, len(order))
        metrics["kl_ce_ratio"] = (
            metrics["kl_mean"] / metrics["ce_mean"] if metrics["ce_mean"] else 0.0
        )
        metrics["alpha"] = {t: float(a.detach()) for t, a in alphas.items()}
        history.append(metrics)
        print(f"  val {json.dumps(metrics)}", flush=True)
        if best is None or metrics["corrected"] > best:
            best = metrics["corrected"]
            stale = 0
            best_state = {
                "heads": {k: v.detach().cpu() for k, v in model.state_dict().items()},
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
                "rank": args.rank,
                "max_length": args.max_length,
                "kl_weight": args.kl_weight,
                "ce_weight": args.ce_weight,
                "seed": args.seed,
            },
            "history": history,
        },
        args.output / "residual.pt",
    )
    (args.output / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {args.output/'residual.pt'} best_cc={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
