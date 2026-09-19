"""Calibration-aware training loop.

Extends the supervised training loop with composite losses:
    L = alpha * task_loss + beta * calibration_loss

where task_loss is the standard CE/BCE and calibration_loss is one of
Brier, MMCE, or focal (from losses.py). Focal replaces the task loss
entirely since it is itself a classification loss.

Usage:
    from model.training.calibration import compute_calibrated_loss, train

    # In training config:
    loss_cfg = {"name": "brier", "alpha": 0.7, "beta": 0.3}
    # or
    loss_cfg = {"name": "focal", "gamma": 2.0, "label_smoothing": 0.05}
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.training.losses import (
    brier_loss,
    brier_loss_binary,
    focal_loss,
    focal_loss_binary,
    mmce_loss,
    mmce_loss_binary,
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationLossConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _compute_task_and_cal_loss(
    q_type: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: CalibrationLossConfig,
) -> torch.Tensor:
    """Compute composite loss for a single item.

    For focal: replaces the task loss entirely (focal IS a task loss).
    For brier/mmce: task loss (CE) + calibration term.
    """
    is_binary = q_type == "noul"

    if cfg.name == "focal":
        if is_binary:
            return focal_loss_binary(logits, target, gamma=cfg.gamma)
        return focal_loss(
            logits, target, gamma=cfg.gamma, label_smoothing=cfg.label_smoothing
        )

    # task loss: standard CE / BCE
    if is_binary:
        task = F.binary_cross_entropy_with_logits(logits, target)
    else:
        task = F.cross_entropy(logits, target)

    # calibration term
    if cfg.name == "brier":
        if is_binary:
            cal = brier_loss_binary(logits, target)
        else:
            cal = brier_loss(logits, target)
    elif cfg.name == "mmce":
        if is_binary:
            cal = mmce_loss_binary(logits, target, kernel_bandwidth=cfg.mmce_bandwidth)
        else:
            cal = mmce_loss(logits, target, kernel_bandwidth=cfg.mmce_bandwidth)
    else:
        raise ValueError(f"Unknown calibration loss: {cfg.name}")

    return cfg.alpha * task + cfg.beta * cal


def compute_calibrated_loss(
    model: nn.Module,
    item: Any,
    cfg: CalibrationLossConfig,
) -> torch.Tensor | None:
    """Compute calibration-aware loss for a single typed-question item.

    Drop-in replacement for supervised.compute_loss with calibration terms.
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
        return _compute_task_and_cal_loss("noul", logit, target, cfg)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if label not in keys:
            return None
        target_idx = keys.index(label)
        target = torch.tensor([target_idx], device=logits.device)
        return _compute_task_and_cal_loss("choice", logits, target, cfg)

    elif q_type == "score":
        criteria = question["criteria"]
        logits = model.forward_score(state, instructions, criteria)
        target_idx = int(round(float(label)))
        target_idx = max(0, min(len(criteria) - 1, target_idx))
        target = torch.tensor([target_idx], device=logits.device)
        return _compute_task_and_cal_loss("score", logits, target, cfg)

    return None


def train_epoch(
    model: nn.Module,
    train_items: list,
    optimizer: torch.optim.Optimizer,
    cfg: CalibrationLossConfig,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with calibration-aware loss."""
    model.train()
    total_loss = 0.0
    n_items = 0
    n_skipped = 0

    for item in train_items:
        optimizer.zero_grad()
        loss = compute_calibrated_loss(model, item, cfg)
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
    """Full calibration-aware training loop.

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

    print(f"Calibration training: loss={cfg.name} alpha={cfg.alpha} beta={cfg.beta}")
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
