"""Report each type'"'"'s pool against its per-epoch sampling slots.

Lives outside ``model/scripts/train_multitask.py`` so that it can be tested on
an interpreter without torch, which is what CI runs.
"""

from __future__ import annotations

from collections import Counter


def print_pool_report(
    pool_by_type: Counter[str],
    sampling: dict,
) -> None:
    """Print how often each type's pool is resampled within one epoch.

    The training mixture is defined by ``type_ratios`` over a fixed
    ``epoch_size``, not by the pool sizes.  When one type's pool is much
    smaller than its share of the epoch, its items are resampled far more
    often than the others — less diverse supervision at a higher overfitting
    risk — and nothing else in the log makes that visible.
    """
    ratios = sampling.get("difficulty_weights") or sampling.get("type_ratios") or {}
    epoch_size = sampling.get("epoch_size")
    if not ratios or not epoch_size:
        return

    # The sampler distributes an epoch over the types that actually have items
    # (``Sampler._active_types``), so an empty type takes no share and the
    # remaining types split the epoch between them.  Dividing by every ratio
    # here would understate every other type's slots -- by a third in the
    # two-type case the table exists to reveal.
    active = [q_type for q_type in ratios if pool_by_type.get(q_type, 0)]
    total_ratio = sum(ratios[q_type] for q_type in active)
    print("\n  Pool vs per-epoch sampling:", flush=True)
    print(
        f"  {'Type':<8} {'Pool':>8} {'Slots':>8} {'Resampling':>11}",
        flush=True,
    )
    print(f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 11}", flush=True)
    for q_type in sorted(ratios):
        pool = pool_by_type.get(q_type, 0)
        slots = int(epoch_size * ratios[q_type] / total_ratio) if total_ratio else 0
        repeats = f"{slots / pool:.2f}x" if pool else "n/a"
        print(
            f"  {q_type:<8} {pool:>8,} {slots:>8,} {repeats:>11}",
            flush=True,
        )
