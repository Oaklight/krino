"""Supervised training loop for decision heads.

Phase 1 training: cross-entropy loss on gold labels with frozen backbone.
Only head parameters receive gradients.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def compute_loss(
    model: nn.Module,
    item: Any,
) -> torch.Tensor | None:
    """Compute loss for a single typed-question item."""
    state = item.state
    question = item.question
    label = item.label
    q_type = question.get("type", "noul")
    instructions = question.get("instructions", "")

    if q_type == "noul":
        logit = model.forward_noul(state, instructions)
        target = torch.tensor([[1.0 if label else 0.0]], device=logit.device)
        return F.binary_cross_entropy_with_logits(logit, target)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if label not in keys:
            return None
        target_idx = keys.index(label)
        target = torch.tensor([target_idx], device=logits.device)
        return F.cross_entropy(logits, target)

    elif q_type == "score":
        criteria = question["criteria"]
        level_texts = criteria
        logits = model.forward_score(state, instructions, level_texts)
        target_idx = int(round(float(label)))
        target_idx = max(0, min(len(criteria) - 1, target_idx))
        target = torch.tensor([target_idx], device=logits.device)
        return F.cross_entropy(logits, target)

    return None


def train_epoch(
    model: nn.Module,
    train_items: list,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch. Returns loss statistics."""
    model.train()
    total_loss = 0.0
    n_items = 0
    n_skipped = 0

    for item in train_items:
        optimizer.zero_grad()
        loss = compute_loss(model, item)
        if loss is None:
            n_skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()
        total_loss += loss.item()
        n_items += 1

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
        loss = compute_loss(model, item)
        if loss is None:
            continue
        total_loss += loss.item()
        n_items += 1

        answer = model.predict(item.state, item.question)
        q_type = item.question.get("type", "noul")
        if q_type == "noul":
            pred = answer.get("noul", 0.5) > 0.5
            if pred == item.label:
                correct += 1
        elif q_type == "choice":
            if answer.get("choice") == item.label:
                correct += 1
        elif q_type == "score":
            pred_level = int(round(answer.get("score", 0)))
            if pred_level == int(round(float(item.label))):
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
) -> dict[str, Any]:
    """Full training loop with evaluation and checkpointing."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training: {sum(p.numel() for p in trainable)} trainable params")
    print(f"  {len(train_items)} train items, {len(eval_items)} eval items, {epochs} epochs")

    best_eval_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        import random
        shuffled = list(train_items)
        random.shuffle(shuffled)

        train_stats = train_epoch(model, shuffled, optimizer, max_grad_norm)
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
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    head_state = {k: v for k, v in model.state_dict().items() if "backbone" not in k}
                    torch.save(head_state, checkpoint_dir / "best_heads.pt")
        else:
            print(f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)")

        history.append(log)

    if checkpoint_dir:
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"history": history, "best_eval_loss": best_eval_loss}
