"""Supervised training loop for decision heads.

Phase 1 training: cross-entropy loss on gold labels with frozen backbone.
Only head parameters receive gradients.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def _score_target(label: Any, n_levels: int, device: torch.device) -> torch.Tensor:
    """Build a score target distribution.

    Integer labels produce one-hot targets. Float labels interpolate between
    adjacent levels (e.g. 2.5 → [0, 0, 0.5, 0.5, 0]).
    """
    label_f = float(label)
    label_f = max(0.0, min(float(n_levels - 1), label_f))
    if label_f == int(label_f):
        return F.one_hot(torch.tensor(int(label_f), device=device), n_levels).float()
    lo = int(label_f)
    hi = min(lo + 1, n_levels - 1)
    frac = label_f - lo
    target = torch.zeros(n_levels, device=device)
    target[lo] = 1.0 - frac
    target[hi] = frac
    return target


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target distribution."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def _teacher_loss(logits: torch.Tensor, teacher_probs: dict, keys: list[str]) -> torch.Tensor:
    """KL-divergence against Jev teacher probability distribution."""
    teacher = torch.tensor([teacher_probs.get(k, 0.0) for k in keys], device=logits.device)
    teacher = teacher / teacher.sum().clamp(min=1e-9)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, teacher.unsqueeze(0), reduction="batchmean")


def compute_loss(
    model: nn.Module,
    item: Any,
) -> torch.Tensor | None:
    """Compute loss for a single typed-question item.

    Supports three loss modes:
    - Standard: one-hot CE (noul→BCE, choice→CE, score→CE or interpolated soft CE)
    - Teacher distillation: KL-divergence against item.teacher_probs when available
    - Float score labels: interpolated soft targets between adjacent levels
    """
    state = item.state
    question = item.question
    label = item.label
    q_type = question.get("type", "noul")
    instructions = question.get("instructions", "")
    teacher_probs = getattr(item, "teacher_probs", None)

    if q_type == "noul":
        logit = model.forward_noul(state, instructions)
        if isinstance(label, str):
            label = label.lower() in ("true", "yes", "1")
        if teacher_probs and "noul" in teacher_probs:
            teacher_p = float(teacher_probs["noul"])
            target = torch.tensor([[teacher_p]], device=logit.device)
        else:
            target = torch.tensor([[1.0 if label else 0.0]], device=logit.device)
        return F.binary_cross_entropy_with_logits(logit, target)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if teacher_probs:
            return _teacher_loss(logits, teacher_probs, keys)
        if label not in keys:
            return None
        target_idx = keys.index(label)
        target = torch.tensor([target_idx], device=logits.device)
        return F.cross_entropy(logits, target)

    elif q_type == "score":
        criteria = question["criteria"]
        level_texts = criteria
        logits = model.forward_score(state, instructions, level_texts)
        n_levels = len(criteria)
        if teacher_probs:
            level_keys = [str(i) for i in range(n_levels)]
            return _teacher_loss(logits, teacher_probs, level_keys)
        label_f = float(label)
        target = _score_target(label, n_levels, logits.device)
        if label_f == int(label_f):
            return F.cross_entropy(logits, target.argmax().unsqueeze(0))
        return _soft_cross_entropy(logits, target.unsqueeze(0))

    return None


def train_epoch(
    model: nn.Module,
    train_items: list,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    """Train for one epoch. Returns loss statistics.

    Args:
        accumulation_steps: number of items to accumulate gradients over
            before an optimizer step. Default 1 preserves per-item SGD.
    """
    model.train()
    total_loss = 0.0
    n_items = 0
    n_skipped = 0
    accum_count = 0

    optimizer.zero_grad()

    for item in train_items:
        loss = compute_loss(model, item)
        if loss is None:
            n_skipped += 1
            continue

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        total_loss += loss.item()
        n_items += 1
        accum_count += 1

        if accum_count >= accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    if accum_count > 0:
        # Re-scale so the partial batch has the same effective LR as full batches
        if accum_count < accumulation_steps:
            scale = accumulation_steps / accum_count
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    return {
        "mean_loss": total_loss / max(n_items, 1),
        "n_items": n_items,
        "n_skipped": n_skipped,
    }


@torch.no_grad()
def eval_epoch(model: nn.Module, eval_items: list) -> dict[str, Any]:
    """Evaluate on a set of items. Returns loss and accuracy statistics."""
    model.eval()
    total_loss = 0.0
    n_items = 0
    correct = 0

    for item in eval_items:
        result = _eval_item(model, item)
        if result is None:
            continue
        loss_val, is_correct = result
        total_loss += loss_val
        n_items += 1
        if is_correct:
            correct += 1

    return {
        "mean_loss": total_loss / max(n_items, 1),
        "accuracy": correct / max(n_items, 1),
        "n_items": n_items,
        "correct": correct,
    }


def train(
    model: nn.Module,
    train_items: list,
    eval_items: list,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Path | None = None,
    eval_every: int = 1,
    accumulation_steps: int = 1,
) -> dict[str, Any]:
    """Full training loop with evaluation and checkpointing."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    print(f"Training: {sum(p.numel() for p in trainable)} trainable params")
    print(f"  {len(train_items)} train items, {len(eval_items)} eval items, {epochs} epochs")

    best_eval_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        shuffled = list(train_items)
        random.shuffle(shuffled)

        train_stats = train_epoch(model, shuffled, optimizer, max_grad_norm, accumulation_steps)
        scheduler.step()

        elapsed = time.monotonic() - t0

        log = {"epoch": epoch, "train": train_stats, "elapsed_s": round(elapsed, 2), "lr": scheduler.get_last_lr()[0]}

        if epoch % eval_every == 0 or epoch == epochs:
            eval_stats = eval_epoch(model, eval_items)
            log["eval"] = eval_stats
            print(
                f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} "
                f"eval_loss={eval_stats['mean_loss']:.4f} eval_acc={eval_stats['accuracy']:.4f} "
                f"({elapsed:.1f}s)"
            )

            if eval_stats["mean_loss"] < best_eval_loss:
                best_eval_loss = eval_stats["mean_loss"]
                log["best"] = True
                if checkpoint_dir:
                    model.save_heads(checkpoint_dir / "best_heads.pt")
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_eval_loss": best_eval_loss,
                    }, checkpoint_dir / "training_state.pt")
        else:
            print(f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)")

        history.append(log)

    if checkpoint_dir:
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"history": history, "best_eval_loss": best_eval_loss}


def _eval_item(
    model: nn.Module,
    item: Any,
) -> tuple[float, bool] | None:
    """Evaluate a single item, returning (loss, is_correct).

    Shared logic for eval_epoch and eval_epoch_detailed.
    Returns None if the item was skipped (e.g. label not in option keys).
    """
    state = item.state
    question = item.question
    label = item.label
    q_type = question.get("type", "noul")
    instructions = question.get("instructions", "")

    if q_type == "noul":
        if isinstance(label, str):
            label = label.lower() in ("true", "yes", "1")
        logit = model.forward_noul(state, instructions)
        target = torch.tensor([[1.0 if label else 0.0]], device=logit.device)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        pred = torch.sigmoid(logit).item() > 0.5
        return (loss.item(), pred == label)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if label not in keys:
            return None
        target_idx = keys.index(label)
        target = torch.tensor([target_idx], device=logits.device)
        loss = F.cross_entropy(logits, target)
        pred_idx = logits.argmax(dim=-1).item()
        return (loss.item(), keys[pred_idx] == label)

    elif q_type == "score":
        criteria = question["criteria"]
        n_levels = len(criteria)
        logits = model.forward_score(state, instructions, criteria)
        label_f = float(label)
        target = _score_target(label, n_levels, logits.device)
        if label_f == int(label_f):
            loss = F.cross_entropy(logits, target.argmax().unsqueeze(0))
        else:
            loss = _soft_cross_entropy(logits, target.unsqueeze(0))
        pred_idx = logits.argmax(dim=-1).item()
        gold_idx = int(round(float(label)))
        gold_idx = max(0, min(n_levels - 1, gold_idx))
        return (loss.item(), pred_idx == gold_idx)

    return None


@torch.no_grad()
def eval_epoch_detailed(model: nn.Module, eval_items: list) -> dict[str, Any]:
    """Evaluate with per-type and per-source breakdowns.

    Uses the same per-item logic as eval_epoch (forward pass + loss + prediction),
    but accumulates into buckets keyed by question type and source.

    Returns:
        Dict with keys:
        - "aggregate": {"mean_loss", "accuracy", "n_items", "correct"}
        - "by_type": {type_name: {"mean_loss", "accuracy", "n_items", "correct"}}
        - "by_source": {source_name: {"mean_loss", "accuracy", "n_items", "correct", "type"}}
    """
    model.eval()

    # Aggregate accumulators
    total_loss = 0.0
    n_items = 0
    correct = 0

    # Per-type accumulators
    type_stats: dict[str, dict[str, float]] = {}
    # Per-source accumulators
    source_stats: dict[str, dict[str, Any]] = {}

    for item in eval_items:
        result = _eval_item(model, item)
        if result is None:
            continue

        loss_val, is_correct = result
        q_type = item.question.get("type", "noul")
        source = item.source

        # Aggregate
        total_loss += loss_val
        n_items += 1
        if is_correct:
            correct += 1

        # Per-type
        if q_type not in type_stats:
            type_stats[q_type] = {"total_loss": 0.0, "n_items": 0, "correct": 0}
        type_stats[q_type]["total_loss"] += loss_val
        type_stats[q_type]["n_items"] += 1
        if is_correct:
            type_stats[q_type]["correct"] += 1

        # Per-source
        if source not in source_stats:
            source_stats[source] = {"total_loss": 0.0, "n_items": 0, "correct": 0, "type": q_type}
        source_stats[source]["total_loss"] += loss_val
        source_stats[source]["n_items"] += 1
        if is_correct:
            source_stats[source]["correct"] += 1

    # Build result dicts
    aggregate = {
        "mean_loss": total_loss / max(n_items, 1),
        "accuracy": correct / max(n_items, 1),
        "n_items": n_items,
        "correct": correct,
    }

    by_type = {}
    for t, s in type_stats.items():
        n = s["n_items"]
        by_type[t] = {
            "mean_loss": s["total_loss"] / max(n, 1),
            "accuracy": s["correct"] / max(n, 1),
            "n_items": n,
            "correct": int(s["correct"]),
        }

    by_source = {}
    for src, s in source_stats.items():
        n = s["n_items"]
        by_source[src] = {
            "mean_loss": s["total_loss"] / max(n, 1),
            "accuracy": s["correct"] / max(n, 1),
            "n_items": n,
            "correct": int(s["correct"]),
            "type": s["type"],
        }

    return {
        "aggregate": aggregate,
        "by_type": by_type,
        "by_source": by_source,
    }


def train_multitask(
    model: nn.Module,
    train_items: list,
    eval_items: list,
    sampler_config: "SamplerConfig",
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Path | None = None,
    eval_every: int = 1,
    accumulation_steps: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Multi-task training loop with type-balanced sampling and detailed eval.

    Same structure as train() but:
    - Uses MultitaskSampler for epoch ordering instead of random.shuffle
    - Calls eval_epoch_detailed() for per-type/per-source breakdowns
    - Prints per-type accuracy summary at each eval checkpoint
    - Uses aggregate mean_loss for best-eval checkpointing

    Args:
        sampler_config: SamplerConfig for MultitaskSampler.
        seed: Base seed for the sampler.
    """
    from data.sampler import MultitaskSampler

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    sampler = MultitaskSampler(train_items, sampler_config, seed=seed)

    print(f"Multi-task training: {sum(p.numel() for p in trainable)} trainable params")
    print(f"  {len(train_items)} train items, {len(eval_items)} eval items")
    print(f"  {epochs} epochs, epoch_size={sampler.epoch_size}, accumulation_steps={accumulation_steps}")

    best_eval_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        epoch_items = sampler.sample_epoch(epoch - 1)

        train_stats = train_epoch(model, epoch_items, optimizer, max_grad_norm, accumulation_steps)
        scheduler.step()

        elapsed = time.monotonic() - t0

        log: dict[str, Any] = {
            "epoch": epoch,
            "train": train_stats,
            "elapsed_s": round(elapsed, 2),
            "lr": scheduler.get_last_lr()[0],
        }

        if epoch % eval_every == 0 or epoch == epochs:
            eval_stats = eval_epoch_detailed(model, eval_items)
            log["eval"] = eval_stats

            agg = eval_stats["aggregate"]
            print(
                f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} "
                f"eval_loss={agg['mean_loss']:.4f} eval_acc={agg['accuracy']:.4f} "
                f"({elapsed:.1f}s)"
            )

            # Per-type summary
            by_type = eval_stats.get("by_type", {})
            type_parts = []
            for t in sorted(by_type):
                ts = by_type[t]
                type_parts.append(f"{t}={ts['accuracy']:.3f}({ts['n_items']})")
            if type_parts:
                print(f"    by_type: {' | '.join(type_parts)}")

            if agg["mean_loss"] < best_eval_loss:
                best_eval_loss = agg["mean_loss"]
                log["best"] = True
                if checkpoint_dir:
                    model.save_heads(checkpoint_dir / "best_heads.pt")
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_eval_loss": best_eval_loss,
                        },
                        checkpoint_dir / "training_state.pt",
                    )
        else:
            print(f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)")

        history.append(log)

    if checkpoint_dir:
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))

    return {"history": history, "best_eval_loss": best_eval_loss}
