"""Torch-free sharding arithmetic for data-parallel training.

Split out of :mod:`model.training.distributed` so the invariants that keep ranks
aligned can be tested on an interpreter without torch — CI installs only pytest.

The invariant that matters: the training loop issues one all-reduce per optimizer
step, and a collective has no timeout. Two ranks that take different numbers of
steps therefore deadlock the group. Step count is a function of the shard
length, so shards must be equal length, and shard length must not depend on
anything a rank decides at run time.
"""

from __future__ import annotations

from typing import Any, Sequence


def shard_length(n_items: int, world_size: int) -> int:
    """Items per rank after padding: ``ceil(n_items / world_size)``."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return -(-n_items // world_size)


def steps_for(shard_len: int, accumulation_steps: int) -> int:
    """Optimizer steps a rank takes over ``shard_len`` items."""
    if accumulation_steps <= 0:
        raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}")
    return -(-shard_len // accumulation_steps)


def shard(items: Sequence[Any], rank: int, world_size: int) -> list[Any]:
    """Return this rank's stride slice of ``items``, padded to equal length.

    A stride keeps the union of all slices equal to the input and the slices
    within one item of each other; equal length matters more than exact
    coverage, because the collective sequence is derived from it. When
    ``len(items)`` is not a multiple of ``world_size`` the shorter slices repeat
    their first item, adding at most ``world_size - 1`` item-steps per epoch
    across the whole group.

    Args:
        items: Items to split.
        rank: This process's rank, in ``[0, world_size)``.
        world_size: Number of processes.

    Returns:
        Exactly ``shard_length(len(items), world_size)`` items.

    Raises:
        ValueError: If a rank would receive no items at all, which means the
            group is larger than the data.
    """
    if world_size <= 1:
        return list(items)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")

    longest = shard_length(len(items), world_size)
    if longest == 0:
        # An empty epoch is legitimate: every rank gets nothing, every rank takes
        # zero steps, so the collective sequence still matches.
        return []

    slice_ = list(items[rank::world_size])
    if not slice_:
        raise ValueError(
            f"cannot shard {len(items)} items across {world_size} ranks: "
            f"rank {rank} would receive none"
        )
    while len(slice_) < longest:
        slice_.append(slice_[0])
    return slice_
