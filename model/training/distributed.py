"""Distributed-data-parallel helpers for the training loop.

The training loop calls the model's own forward methods (``forward_noul``,
``forward_choice``, ``forward_score``) instead of ``forward()``, so
``torch.nn.parallel.DistributedDataParallel`` cannot wrap it: DDP arms its
gradient reducer inside ``DDP.forward``, which those calls never reach. The
loop therefore averages gradients explicitly at each optimizer step. That is
what DDP does internally, minus the overlapped prefetch, and it works with any
call pattern.

Two behaviours are worth knowing:

* Every parameter that requires grad is averaged, with a zero gradient for the
  parameters a rank did not touch in that step. A rank's step therefore covers
  the same parameter set as every other rank's, which keeps replicas identical
  after ``optimizer.step()``. It also means an unused head still receives its
  weight decay, as it would under DDP.
* Each rank trains on a stride-sharded slice of the epoch the sampler
  produced. The global item composition is unchanged, but one optimizer step
  now spans ``world_size x accumulation_steps`` items unless
  ``accumulation_steps`` is divided by ``world_size``.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch


def world_info() -> tuple[int, int, int]:
    """Read rank, world size, and local rank from the launcher environment.

    Returns:
        ``(rank, world_size, local_rank)``; ``(0, 1, 0)`` outside a
        ``torchrun``/``mp.spawn`` launch.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def init_process_group(local_rank: int = 0) -> tuple[int, int]:
    """Initialise the default process group when launched distributed.

    Args:
        local_rank: Index of the GPU this process owns on its node.

    Returns:
        ``(rank, world_size)``.
    """
    rank, world_size, local_rank = world_info()
    if world_size <= 1:
        return rank, world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    torch.distributed.init_process_group(backend=backend)
    return rank, world_size


def is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main_process() -> bool:
    """True on rank 0, which owns logging, evaluation output, and checkpoints."""
    return world_info()[0] == 0


def barrier() -> None:
    if is_initialized():
        torch.distributed.barrier()


def shutdown() -> None:
    if is_initialized():
        torch.distributed.destroy_process_group()


def shard(items: Sequence[Any], rank: int, world_size: int) -> list[Any]:
    """Return this rank's stride slice of ``items``.

    A stride keeps the union of all slices equal to the input and the slices
    near-equal in size, and it is deterministic for a given epoch.

    Args:
        items: Items to split.
        rank: This process's rank.
        world_size: Number of processes.

    Returns:
        The items this rank should process.
    """
    if world_size <= 1:
        return list(items)
    return list(items[rank::world_size])


def average_gradients(params: Sequence[torch.nn.Parameter], world_size: int) -> None:
    """All-reduce ``params`` gradients so every replica sees the global mean.

    Parameters with no gradient on this rank are given a zero gradient first,
    so that all ranks call the collective over the same tensors. The zero
    gradient is not removed afterwards: the optimizer then treats an unused
    parameter exactly as it would after a DDP step.

    Gradients are concatenated per dtype so a step costs one collective, which
    matters because the loop steps per accumulated item rather than per large
    batch.

    Args:
        params: Parameters that require grad.
        world_size: Number of processes in the group.
    """
    if world_size <= 1 or not is_initialized():
        return
    grads: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grads.append(param.grad)

    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for grad in grads:
        by_dtype.setdefault(grad.dtype, []).append(grad)

    for group in by_dtype.values():
        flat = torch.cat([grad.reshape(-1) for grad in group])
        torch.distributed.all_reduce(
            flat, op=torch.distributed.ReduceOp.SUM
        )
        flat.div_(world_size)
        offset = 0
        for grad in group:
            size = grad.numel()
            grad.copy_(flat[offset:offset + size].view_as(grad))
            offset += size


def _merge_bucket(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-rank buckets that carry ``n_items``, ``correct``, ``mean_loss``."""
    n_items = sum(b["n_items"] for b in buckets)
    correct = sum(b["correct"] for b in buckets)
    total_loss = sum(b["mean_loss"] * b["n_items"] for b in buckets)
    merged = dict(buckets[0])
    merged.update(
        {
            "mean_loss": total_loss / max(n_items, 1),
            "accuracy": correct / max(n_items, 1),
            "n_items": n_items,
            "correct": int(correct),
        }
    )
    return merged


def merge_eval_stats(stats_list: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine ``eval_epoch_detailed`` results computed on disjoint item sets.

    Args:
        stats_list: One result dict per rank, each over that rank's shard.

    Returns:
        A result dict with the same shape, weighted by item count.
    """
    if not stats_list:
        return {"aggregate": {"mean_loss": 0.0, "accuracy": 0.0, "n_items": 0, "correct": 0},
                "by_type": {}, "by_source": {}}
    if len(stats_list) == 1:
        return stats_list[0]

    merged: dict[str, Any] = {
        "aggregate": _merge_bucket([s["aggregate"] for s in stats_list]),
        "by_type": {},
        "by_source": {},
    }
    for key in ("by_type", "by_source"):
        names: list[str] = []
        for stats in stats_list:
            for name in stats.get(key, {}):
                if name not in names:
                    names.append(name)
        for name in names:
            buckets = [s[key][name] for s in stats_list if name in s.get(key, {})]
            merged[key][name] = _merge_bucket(buckets)
    return merged


def gather_train_stats(stats: dict[str, float], world_size: int) -> dict[str, float]:
    """Combine ``train_epoch`` statistics computed on disjoint item shards.

    Args:
        stats: This rank's ``{"mean_loss", "n_items", "n_skipped"}``.
        world_size: Number of data-parallel processes.

    Returns:
        Item-weighted statistics over all ranks, so the logged loss is
        comparable with a single-process run.
    """
    if world_size <= 1 or not is_initialized():
        return stats
    gathered: list[dict[str, float] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, stats)
    items = [s for s in gathered if s is not None]
    n_items = sum(s["n_items"] for s in items)
    total_loss = sum(s["mean_loss"] * s["n_items"] for s in items)
    return {
        "mean_loss": total_loss / max(n_items, 1),
        "n_items": n_items,
        "n_skipped": sum(s["n_skipped"] for s in items),
    }


def gather_eval_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Collect every rank's eval stats and return the merged result.

    All ranks receive the same merged dict, so logging and checkpoint decisions
    stay consistent without an extra broadcast.

    Args:
        stats: This rank's ``eval_epoch_detailed`` result.

    Returns:
        Merged result over all ranks.
    """
    _, world_size, _ = world_info()
    if world_size <= 1 or not is_initialized():
        return stats
    gathered: list[dict[str, Any] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, stats)
    return merge_eval_stats([s for s in gathered if s is not None])
