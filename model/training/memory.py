"""Phase-scoped device-memory snapshots, kept free of torch so they can be tested.

The question issue #161 asks — did the 8B run *grow* across epochs, or was it
fragmentation? — depends on two things the first version of this logging got
wrong:

* a peak must be paired with a peak. ``max_memory_allocated()`` is a peak over
  the phase but ``memory_reserved()`` is the value at that instant, so the two
  halves of the pair are sampled at different moments and the ratio can come out
  with reserved below allocated.
* training and eval must not be charged to each other. A single snapshot taken
  before the eval pass attributes eval's peak to the *next* epoch, so every epoch
  after the first is inflated by its predecessor's eval — which is exactly the
  shape that makes a heavier eval look like growth.

This module only does the arithmetic; callers pass the torch functions in. That
keeps the logic testable on an interpreter without torch, which is what CI has.
"""

from __future__ import annotations

from typing import Any, Callable

_GIB = 1024**3


def peak_snapshot(
    max_allocated: Callable[[], int],
    max_reserved: Callable[[], int],
    stats: Callable[[], dict[str, Any]],
    reset: Callable[[], None],
) -> dict[str, float | int]:
    """Read this phase's memory peaks, then reset the counters for the next phase.

    Args:
        max_allocated: Returns the peak allocated bytes since the last reset.
        max_reserved: Returns the peak reserved bytes since the last reset.
        stats: Returns ``torch.cuda.memory_stats()``; ``num_alloc_retries`` is a
            cumulative counter and is reported as such.
        reset: Resets the peak counters. Called after reading, so the next phase
            starts from zero.

    Returns:
        ``peak_allocated`` and ``peak_reserved`` in GiB, their difference as
        ``fragmentation``, and ``alloc_retries_total``. The difference is the
        number #161 is about: a peak that keeps climbing in ``peak_allocated`` is
        growth, one that climbs only in ``fragmentation`` is the allocator
        holding freed blocks.
    """
    allocated = max_allocated() / _GIB
    reserved = max_reserved() / _GIB
    retries = int(stats().get("num_alloc_retries", 0))
    reset()
    return {
        "peak_allocated": round(allocated, 2),
        "peak_reserved": round(reserved, 2),
        # Never negative in theory; rounding can make it -0.0, which reads as a
        # logger bug rather than as a tight fit.
        "fragmentation": round(max(reserved - allocated, 0.0), 2),
        "alloc_retries_total": retries,
    }


def format_epoch_memory(train: dict | None, eval_: dict | None) -> str:
    """One-line rendering of a phase pair, empty when memory is not tracked."""
    if not train:
        return ""
    text = (
        f" mem train={train['peak_allocated']:.1f}/{train['peak_reserved']:.1f}G"
    )
    if eval_:
        text += f" eval={eval_['peak_allocated']:.1f}/{eval_['peak_reserved']:.1f}G"
    retries = int(train.get("alloc_retries_total", 0))
    if retries or (eval_ and int(eval_.get("alloc_retries_total", 0)) > retries):
        # Cumulative since process start: any retry means the allocator had to
        # defragment, which is the condition that precedes a hard OOM.
        text += f" alloc_retries={retries}"
    return text
