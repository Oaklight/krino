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


def _apply_uncertainty_weight(
    loss: torch.Tensor,
    log_var: nn.Parameter,
) -> torch.Tensor:
    """Apply homoscedastic uncertainty weighting to a loss (Kendall et al. 2018).

    Computes: loss / (2 * exp(log_var)) + log_var / 2

    The learned log_var acts as a task-specific uncertainty parameter that
    automatically balances loss magnitudes across tasks.

    Args:
        loss: Scalar loss tensor for a single item.
        log_var: Learnable log-variance parameter for this task type.

    Returns:
        Uncertainty-weighted loss tensor.
    """
    return loss / (2 * torch.exp(log_var)) + log_var / 2


def compute_loss(
    model: nn.Module,
    item: Any,
    log_vars: dict[str, nn.Parameter] | None = None,
) -> torch.Tensor | None:
    """Compute loss for a single typed-question item.

    Supports three loss modes:
    - Standard: one-hot CE (noul→BCE, choice→CE, score→CE or interpolated soft CE)
    - Teacher distillation: KL-divergence against item.teacher_probs when available
    - Float score labels: interpolated soft targets between adjacent levels

    When log_vars is provided, applies homoscedastic uncertainty weighting
    (Kendall et al. 2018) to the computed loss using the type-specific log_var.
    """
    state = item.state
    question = item.question
    label = item.label
    q_type = question.get("type", "noul")
    instructions = question.get("instructions", "")
    teacher_probs = getattr(item, "teacher_probs", None)

    loss: torch.Tensor | None = None

    if q_type == "noul":
        logit = model.forward_noul(state, instructions)
        if isinstance(label, str):
            label = label.lower() in ("true", "yes", "1")
        if teacher_probs and "noul" in teacher_probs:
            teacher_p = float(teacher_probs["noul"])
            target = torch.tensor([[teacher_p]], device=logit.device)
        else:
            target = torch.tensor([[1.0 if label else 0.0]], device=logit.device)
        loss = F.binary_cross_entropy_with_logits(logit, target)

    elif q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)
        if teacher_probs:
            loss = _teacher_loss(logits, teacher_probs, keys)
        elif label not in keys:
            return None
        else:
            target_idx = keys.index(label)
            target = torch.tensor([target_idx], device=logits.device)
            loss = F.cross_entropy(logits, target)

    elif q_type == "score":
        criteria = question["criteria"]
        level_texts = criteria
        logits = model.forward_score(state, instructions, level_texts)
        n_levels = len(criteria)
        if teacher_probs:
            level_keys = [str(i) for i in range(n_levels)]
            loss = _teacher_loss(logits, teacher_probs, level_keys)
        else:
            label_f = float(label)
            target = _score_target(label, n_levels, logits.device)
            if label_f == int(label_f):
                loss = F.cross_entropy(logits, target.argmax().unsqueeze(0))
            else:
                loss = _soft_cross_entropy(logits, target.unsqueeze(0))

    if loss is not None and log_vars is not None and q_type in log_vars:
        loss = _apply_uncertainty_weight(loss, log_vars[q_type])

    return loss


def _compute_noul_batch(
    model: nn.Module,
    items: list,
) -> list[torch.Tensor]:
    """Batch-compute noul losses: one backbone call for all contexts."""
    texts = [f"{it.state} {it.question.get('instructions', '')}" for it in items]
    pooled = model._encode_text(texts)  # [N, hidden]
    logits = model.noul_head(pooled)  # [N, 1]

    targets = []
    for item in items:
        label = item.label
        if isinstance(label, str):
            label = label.lower() in ("true", "yes", "1")
        teacher_probs = getattr(item, "teacher_probs", None)
        if teacher_probs and "noul" in teacher_probs:
            targets.append(float(teacher_probs["noul"]))
        else:
            targets.append(1.0 if label else 0.0)

    target_t = torch.tensor(targets, device=logits.device).unsqueeze(1)  # [N, 1]
    batch_loss = F.binary_cross_entropy_with_logits(logits, target_t, reduction="none")
    return [batch_loss[i] for i in range(len(items))]


def _compute_choice_batch(
    model: nn.Module,
    items: list,
) -> list[torch.Tensor | None]:
    """Batch-compute choice losses: per-item full-sequence context + batched options."""
    losses: list[torch.Tensor | None] = []
    for item in items:
        criteria = item.question["criteria"]
        keys = list(criteria.keys())
        option_texts = [v or k for k, v in criteria.items()]
        teacher_probs = getattr(item, "teacher_probs", None)

        context_text = f"{item.state} {item.question.get('instructions', '')}"
        context_hidden = model._encode_with_sequence(context_text)  # [1, seq_len, hidden]
        option_pooled = model._encode_text(option_texts)  # [K, hidden]
        opt_hidden = option_pooled.unsqueeze(0)  # [1, K, hidden]
        logits = model.choice_head(context_hidden, opt_hidden)

        if teacher_probs:
            losses.append(_teacher_loss(logits, teacher_probs, keys))
        elif item.label not in keys:
            losses.append(None)
        else:
            target_idx = keys.index(item.label)
            target = torch.tensor([target_idx], device=logits.device)
            losses.append(F.cross_entropy(logits, target))
    return losses


def _compute_score_batch(
    model: nn.Module,
    items: list,
) -> list[torch.Tensor | None]:
    """Batch-compute score losses: per-item full-sequence context + batched levels."""
    losses: list[torch.Tensor | None] = []
    for item in items:
        criteria = item.question["criteria"]
        n_levels = len(criteria)
        teacher_probs = getattr(item, "teacher_probs", None)

        context_text = f"{item.state} {item.question.get('instructions', '')}"
        context_hidden = model._encode_with_sequence(context_text)  # [1, seq_len, hidden]
        level_pooled = model._encode_text(criteria)  # [L, hidden]
        lvl_hidden = level_pooled.unsqueeze(0)  # [1, L, hidden]
        logits = model.score_head(context_hidden, lvl_hidden)

        if teacher_probs:
            level_keys = [str(j) for j in range(n_levels)]
            losses.append(_teacher_loss(logits, teacher_probs, level_keys))
        else:
            label_f = float(item.label)
            target = _score_target(item.label, n_levels, logits.device)
            if label_f == int(label_f):
                losses.append(F.cross_entropy(logits, target.argmax().unsqueeze(0)))
            else:
                losses.append(_soft_cross_entropy(logits, target.unsqueeze(0)))
    return losses


def train_epoch(
    model: nn.Module,
    train_items: list,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
    batch_backbone: int = 0,
    log_vars: dict[str, nn.Parameter] | None = None,
) -> dict[str, float]:
    """Train for one epoch. Returns loss statistics.

    Args:
        accumulation_steps: number of items to accumulate gradients over
            before an optimizer step. Default 1 preserves per-item SGD.
        batch_backbone: if > 0, group items by type and batch backbone
            forward passes in chunks of this size. 0 = per-item (legacy).
        log_vars: per-type learnable log-variance parameters for uncertainty
            weighting (Kendall et al. 2018). None disables weighting.
    """
    model.train()
    total_loss = 0.0
    n_items = 0
    n_skipped = 0
    accum_count = 0

    optimizer.zero_grad()

    if batch_backbone > 0:
        # Group items by type, process in batches
        i = 0
        while i < len(train_items):
            # Collect a window of items, group by type
            window = train_items[i:i + batch_backbone]
            i += len(window)

            noul_items = [it for it in window if it.question.get("type") == "noul"]
            choice_items = [it for it in window if it.question.get("type") == "choice"]
            score_items = [it for it in window if it.question.get("type") == "score"]

            # Compute losses per type, applying uncertainty weighting if enabled
            all_weighted: list[torch.Tensor] = []
            n_window_skipped = 0

            if noul_items:
                noul_losses = _compute_noul_batch(model, noul_items)
                for loss in noul_losses:
                    if loss is not None:
                        if log_vars is not None and "noul" in log_vars:
                            loss = _apply_uncertainty_weight(loss, log_vars["noul"])
                        all_weighted.append(loss.squeeze())
                    else:
                        n_window_skipped += 1

            if choice_items:
                choice_losses = _compute_choice_batch(model, choice_items)
                for loss in choice_losses:
                    if loss is not None:
                        if log_vars is not None and "choice" in log_vars:
                            loss = _apply_uncertainty_weight(loss, log_vars["choice"])
                        all_weighted.append(loss.squeeze())
                    else:
                        n_window_skipped += 1

            if score_items:
                score_losses = _compute_score_batch(model, score_items)
                for loss in score_losses:
                    if loss is not None:
                        if log_vars is not None and "score" in log_vars:
                            loss = _apply_uncertainty_weight(loss, log_vars["score"])
                        all_weighted.append(loss.squeeze())
                    else:
                        n_window_skipped += 1

            n_skipped += n_window_skipped

            # Accumulate valid losses and backward once per window to avoid
            # graph-reuse issues when a trainable projector is shared.
            if all_weighted:
                window_loss = torch.stack(all_weighted).sum()
                scaled_loss = window_loss / accumulation_steps
                scaled_loss.backward()
                total_loss += window_loss.item()
                n_items += len(all_weighted)
                accum_count += len(all_weighted)

            if accum_count >= accumulation_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
    else:
        for item in train_items:
            loss = compute_loss(model, item, log_vars=log_vars)
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
    if mlp_lr is not None and hasattr(model, "projector") and not isinstance(model.projector, nn.Identity):
        projector_params = list(model.projector.parameters())
        projector_ids = {id(p) for p in projector_params}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in projector_ids]
        optimizer = AdamW([
            {"params": head_params, "lr": lr},
            {"params": projector_params, "lr": mlp_lr},
        ], weight_decay=weight_decay)
        print(f"  Per-component LR: heads={lr}, mlp={mlp_lr}", flush=True)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    print(f"Training: {sum(p.numel() for p in trainable)} trainable params", flush=True)
    print(f"  {len(train_items)} train items, {len(eval_items)} eval items, {epochs} epochs", flush=True)

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
            print(f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)", flush=True)

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
    batch_backbone: int = 0,
    save_every_epoch: bool = False,
    mlp_lr: float | None = None,
    noul_lr: float | None = None,
    choice_lr: float | None = None,
    score_lr: float | None = None,
    uncertainty_weighting: bool = False,
) -> dict[str, Any]:
    """Multi-task training loop with type-balanced sampling and detailed eval.

    Same structure as train() but:
    - Uses MultitaskSampler for epoch ordering instead of random.shuffle
    - Calls eval_epoch_detailed() for per-type/per-source breakdowns
    - Prints per-type accuracy summary at each eval checkpoint
    - Uses aggregate mean_loss for best-eval checkpointing
    - Optionally saves per-epoch checkpoints for post-hoc analysis

    Args:
        sampler_config: SamplerConfig for MultitaskSampler.
        seed: Base seed for the sampler.
        save_every_epoch: If True and checkpoint_dir is set, save head weights
            at every eval epoch (epoch_N.pt) in addition to best_heads.pt.
            Cost: ~4MB per save. Enables post-hoc eval on new benchmarks.
        mlp_lr: Separate learning rate for MLP projector. None uses base lr.
        noul_lr: Separate learning rate for noul head. None uses base lr.
        choice_lr: Separate learning rate for choice head. None uses base lr.
        score_lr: Separate learning rate for score head. None uses base lr.
        uncertainty_weighting: If True, learn per-type homoscedastic uncertainty
            parameters (Kendall et al. 2018) to automatically balance losses.
    """
    from data.sampler import MultitaskSampler

    device = next(model.parameters()).device
    use_per_type_lr = any(x is not None for x in [noul_lr, choice_lr, score_lr])
    has_mlp = hasattr(model, "projector") and not isinstance(model.projector, nn.Identity)

    # Build optimizer param groups
    if use_per_type_lr or (mlp_lr is not None and has_mlp):
        # Collect param id sets for each component
        noul_params = list(model.noul_head.parameters())
        choice_params = list(model.choice_head.parameters())
        score_params = list(model.score_head.parameters())
        assigned_ids = {id(p) for p in noul_params + choice_params + score_params}

        if has_mlp:
            projector_params = list(model.projector.parameters())
            assigned_ids.update(id(p) for p in projector_params)

        param_groups = [
            {"params": noul_params, "lr": noul_lr or lr},
            {"params": choice_params, "lr": choice_lr or lr},
            {"params": score_params, "lr": score_lr or lr},
        ]

        if has_mlp:
            param_groups.append({"params": projector_params, "lr": mlp_lr or lr})

        # Remaining trainable params (if any)
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned_ids]
        if remaining:
            param_groups.append({"params": remaining, "lr": lr})

        optimizer = AdamW(param_groups, weight_decay=weight_decay)

        if use_per_type_lr:
            print(
                f"  Per-type LR: noul={noul_lr or lr}, choice={choice_lr or lr}, "
                f"score={score_lr or lr}",
                flush=True,
            )
        if mlp_lr is not None and has_mlp:
            print(f"  Per-component LR: mlp={mlp_lr}", flush=True)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # Setup uncertainty weighting (Kendall et al. 2018)
    log_vars: dict[str, nn.Parameter] | None = None
    if uncertainty_weighting:
        log_vars = {
            "noul": nn.Parameter(torch.zeros(1, device=device)),
            "choice": nn.Parameter(torch.zeros(1, device=device)),
            "score": nn.Parameter(torch.zeros(1, device=device)),
        }
        for lv in log_vars.values():
            optimizer.add_param_group({"params": [lv], "lr": lr})
        print("  Uncertainty weighting: enabled (3 learnable log_var params)", flush=True)

    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    sampler = MultitaskSampler(train_items, sampler_config, seed=seed)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Multi-task training: {trainable_count} trainable params", flush=True)
    print(f"  {len(train_items)} train items, {len(eval_items)} eval items", flush=True)
    print(f"  {epochs} epochs, epoch_size={sampler.epoch_size}, accumulation_steps={accumulation_steps}", flush=True)

    best_eval_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        epoch_items = sampler.sample_epoch(epoch - 1)

        train_stats = train_epoch(
            model, epoch_items, optimizer, max_grad_norm,
            accumulation_steps, batch_backbone, log_vars=log_vars,
        )
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
                print(f"    by_type: {' | '.join(type_parts)}", flush=True)

            # Log uncertainty weights if enabled
            if log_vars is not None:
                lv_parts = []
                lv_snapshot: dict[str, float] = {}
                for t in sorted(log_vars):
                    val = log_vars[t].item()
                    lv_parts.append(f"{t}={val:.4f}")
                    lv_snapshot[t] = round(val, 6)
                print(f"    log_var: {' | '.join(lv_parts)}", flush=True)
                log["log_vars"] = lv_snapshot

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

            if save_every_epoch and checkpoint_dir:
                model.save_heads(checkpoint_dir / f"epoch_{epoch:03d}.pt")
        else:
            print(f"  epoch {epoch}/{epochs}: train_loss={train_stats['mean_loss']:.4f} ({elapsed:.1f}s)", flush=True)

        history.append(log)

    if checkpoint_dir:
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))

    return {"history": history, "best_eval_loss": best_eval_loss}
