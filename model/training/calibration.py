"""Calibration-aware training loop with virtual batching.

Extends the supervised training loop with composite losses:
    L = alpha * task_loss + beta * calibration_loss

where task_loss is the standard CE/BCE and calibration_loss is one of
Brier, MMCE, or focal (from losses.py). Focal replaces the task loss
entirely since it is itself a classification loss.

MMCE requires multiple samples to compute kernel statistics. This module
uses "virtual batching": per-item forward passes (variable-length text,
variable option counts) with batch-level loss computation. Each item's
confidence tensor retains its computation graph so gradients flow back
through the shared head parameters.

Usage:
    from model.training.calibration import train

    train(model, train_items, eval_items,
          loss_cfg={"name": "mmce", "alpha": 0.5, "beta": 0.5, "batch_size": 16})
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.training.losses import (
    focal_loss,
    focal_loss_binary,
    mmce_kernel_loss,
)


@dataclass
class CalibrationLossConfig:
    """Configuration for composite calibration loss."""

    name: str = "brier"
    alpha: float = 0.7
    beta: float = 0.3
    gamma: float = 2.0
    label_smoothing: float = 0.0
    mmce_bandwidth: float = 0.25
    batch_size: int = 16

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationLossConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _ItemResult(NamedTuple):
    """Per-item forward pass result for batched loss computation."""

    task_loss: torch.Tensor
    confidence: torch.Tensor
    correctness: torch.Tensor
    brier_per_item: torch.Tensor


def _forward_and_extract(
    model: nn.Module,
    item: Any,
    cfg: CalibrationLossConfig,
) -> _ItemResult | None:
    """Forward pass for a single item, returning loss components.

    Returns task loss (CE/BCE or focal), confidence in own prediction
    (has grad for MMCE backprop), and correctness indicator (detached).
    """
    state = item.state
    question = item.question
    label = item.label
    q_type = question.get("type", "noul")
    instructions = question.get("instructions", "")

    if q_type == "noul":
        logit = model.forward_noul(state, instructions)
        if isinstance(label, str):
            label = label.lower() in ("true", "yes", "1")
        target = torch.tensor([[1.0 if label else 0.0]], device=logit.device)

        if cfg.name == "focal":
            task_loss = focal_loss_binary(logit, target, gamma=cfg.gamma)
        else:
            task_loss = F.binary_cross_entropy_with_logits(logit, target)

        prob = torch.sigmoid(logit).squeeze()
        confidence = torch.where(prob > 0.5, prob, 1.0 - prob)
        predicted = (prob > 0.5).float()
        correctness = (predicted == target.squeeze()).float().detach()
        brier = (prob - target.squeeze()) ** 2

        return _ItemResult(task_loss, confidence, correctness, brier)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if label not in keys:
            return None
        target_idx = keys.index(label)
        target = torch.tensor([target_idx], device=logits.device)

        if cfg.name == "focal":
            task_loss = focal_loss(
                logits, target, gamma=cfg.gamma, label_smoothing=cfg.label_smoothing
            )
        else:
            task_loss = F.cross_entropy(logits, target)

        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.squeeze()
        predicted = probs.argmax(dim=-1)
        correctness = (predicted.squeeze() == target.squeeze()).float().detach()
        one_hot = F.one_hot(target, num_classes=logits.shape[-1]).float()
        brier = ((probs - one_hot) ** 2).sum(dim=-1).squeeze()

        return _ItemResult(task_loss, confidence, correctness, brier)

    elif q_type == "score":
        # Local import to avoid circular dependency (calibration imports supervised.eval_epoch)
        from .supervised import _score_target, _soft_cross_entropy, _teacher_loss

        criteria = question["criteria"]
        n_levels = len(criteria)
        logits = model.forward_score(state, instructions, criteria)
        teacher_probs = getattr(item, "teacher_probs", None)

        target_dist = _score_target(label, n_levels, logits.device)
        target_idx = int(round(float(label)))
        target_idx = max(0, min(n_levels - 1, target_idx))
        target = torch.tensor([target_idx], device=logits.device)

        label_f = float(label)
        if teacher_probs:
            level_keys = [str(i) for i in range(n_levels)]
            task_loss = _teacher_loss(logits, teacher_probs, level_keys)
        elif cfg.name == "focal":
            task_loss = focal_loss(
                logits, target, gamma=cfg.gamma, label_smoothing=cfg.label_smoothing
            )
        elif label_f == int(label_f):
            task_loss = F.cross_entropy(logits, target)
        else:
            task_loss = _soft_cross_entropy(logits, target_dist.unsqueeze(0))

        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.squeeze()
        predicted = probs.argmax(dim=-1)
        correctness = (predicted.squeeze() == target.squeeze()).float().detach()
        brier = ((probs - target_dist.unsqueeze(0)) ** 2).sum(dim=-1).squeeze()

        return _ItemResult(task_loss, confidence, correctness, brier)

    return None


def _compute_batch_loss(
    results: list[_ItemResult],
    cfg: CalibrationLossConfig,
) -> torch.Tensor:
    """Combine per-item results into a single batch loss."""
    task_loss = torch.stack([r.task_loss for r in results]).mean()

    if cfg.name == "focal":
        return task_loss

    if cfg.name == "mmce":
        if len(results) >= 2:
            confidences = torch.stack([r.confidence for r in results])
            correctness = torch.stack([r.correctness for r in results])
            cal_loss = mmce_kernel_loss(confidences, correctness, cfg.mmce_bandwidth)
        else:
            cal_loss = torch.tensor(0.0, device=task_loss.device)
        return cfg.alpha * task_loss + cfg.beta * cal_loss

    if cfg.name == "brier":
        cal_loss = torch.stack([r.brier_per_item for r in results]).mean()
        return cfg.alpha * task_loss + cfg.beta * cal_loss

    raise ValueError(f"Unknown calibration loss: {cfg.name}")


def train_epoch(
    model: nn.Module,
    train_items: list,
    optimizer: torch.optim.Optimizer,
    cfg: CalibrationLossConfig,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with batched calibration-aware loss."""
    model.train()
    total_loss = 0.0
    n_items = 0
    n_skipped = 0
    n_batches = 0
    batch_size = cfg.batch_size

    for batch_start in range(0, len(train_items), batch_size):
        batch_items = train_items[batch_start : batch_start + batch_size]
        optimizer.zero_grad()

        results: list[_ItemResult] = []
        for item in batch_items:
            result = _forward_and_extract(model, item, cfg)
            if result is None:
                n_skipped += 1
                continue
            results.append(result)

        if not results:
            continue

        loss = _compute_batch_loss(results, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()

        total_loss += loss.item() * len(results)
        n_items += len(results)
        n_batches += 1

    return {
        "mean_loss": total_loss / max(n_items, 1),
        "n_items": n_items,
        "n_skipped": n_skipped,
        "n_batches": n_batches,
    }


def train(
    model: nn.Module,
    train_items: list,
    eval_items: list,
    loss_cfg: dict[str, Any] | CalibrationLossConfig | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Path | None = None,
    eval_every: int = 1,
) -> dict[str, Any]:
    """Full calibration-aware training loop with virtual batching.

    Same interface as supervised.train but with configurable loss.
    Reuses supervised.eval_epoch for evaluation (eval always uses CE
    to keep metrics comparable across loss configurations).
    """
    from model.training.supervised import eval_epoch

    if loss_cfg is None:
        cfg = CalibrationLossConfig()
    elif isinstance(loss_cfg, dict):
        cfg = CalibrationLossConfig.from_dict(loss_cfg)
    else:
        cfg = loss_cfg

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    print(
        f"Calibration training: loss={cfg.name} alpha={cfg.alpha} beta={cfg.beta} "
        f"batch_size={cfg.batch_size}"
    )
    if cfg.name == "focal":
        print(f"  focal gamma={cfg.gamma} label_smoothing={cfg.label_smoothing}")
    elif cfg.name == "mmce":
        print(f"  mmce bandwidth={cfg.mmce_bandwidth}")
    print(
        f"  {sum(p.numel() for p in trainable)} trainable params, "
        f"{len(train_items)} train, {len(eval_items)} eval, {epochs} epochs"
    )

    best_eval_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        shuffled = list(train_items)
        random.shuffle(shuffled)

        train_stats = train_epoch(model, shuffled, optimizer, cfg, max_grad_norm)
        scheduler.step()

        elapsed = time.monotonic() - t0

        log: dict[str, Any] = {
            "epoch": epoch,
            "train": train_stats,
            "elapsed_s": round(elapsed, 2),
            "lr": scheduler.get_last_lr()[0],
            "loss_config": {
                "name": cfg.name,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "batch_size": cfg.batch_size,
            },
        }

        if epoch % eval_every == 0 or epoch == epochs:
            eval_stats = eval_epoch(model, eval_items)
            log["eval"] = eval_stats
            print(
                f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} "
                f"eval_loss={eval_stats['mean_loss']:.4f} "
                f"eval_acc={eval_stats['accuracy']:.4f} ({elapsed:.1f}s)"
            )

            if eval_stats["mean_loss"] < best_eval_loss:
                best_eval_loss = eval_stats["mean_loss"]
                log["best"] = True
                if checkpoint_dir:
                    model.save_heads(checkpoint_dir / "best_heads.pt")
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_eval_loss": best_eval_loss,
                            "loss_config": {
                                "name": cfg.name,
                                "alpha": cfg.alpha,
                                "beta": cfg.beta,
                                "batch_size": cfg.batch_size,
                            },
                        },
                        checkpoint_dir / "training_state.pt",
                    )
        else:
            print(
                f"  epoch {epoch}/{epochs}: "
                f"train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)"
            )

        history.append(log)

    if checkpoint_dir:
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"history": history, "best_eval_loss": best_eval_loss}
