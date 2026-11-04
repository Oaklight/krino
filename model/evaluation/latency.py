"""Latency summary helpers.

Kept free of torch so the arithmetic that turns raw timings into the reported
table can be tested anywhere. The timing loop itself lives in
``model/scripts/bench_latency.py``.
"""

from __future__ import annotations

import statistics

# Option counts that matter for the efficiency target: a binary noul question,
# a small choice set, the 20-option range where decision models are documented
# to degrade, and the published 256-option ceiling.
DEFAULT_OPTION_COUNTS = (2, 5, 20, 256)

_OPTION_STEMS = (
    "Ship now and accept the known failure",
    "Delay the release by one week",
    "Ship behind a feature flag",
    "Roll back to the previous build",
    "Extend the freeze and fix the regression",
)


def make_options(n_options: int) -> list[str]:
    """Build ``n_options`` distinct, similarly shaped candidate strings.

    Args:
        n_options: Number of options to generate.

    Returns:
        Option descriptions; beyond the fifth, a variant suffix keeps them
        distinct so the attention head cannot merge duplicates.

    Raises:
        ValueError: If ``n_options`` is not positive.
    """
    if n_options < 1:
        raise ValueError("n_options must be at least 1")
    options = []
    for idx in range(n_options):
        stem = _OPTION_STEMS[idx % len(_OPTION_STEMS)]
        suffix = f" (variant {idx + 1})" if idx >= len(_OPTION_STEMS) else ""
        options.append(f"{stem}{suffix}")
    return options


def percentile(values: list[float], fraction: float) -> float:
    """Return the ``fraction`` quantile of ``values`` by nearest rank.

    Args:
        values: Samples, in any order.
        fraction: Quantile in ``(0, 1]``.

    Returns:
        The sampled value at that rank.

    Raises:
        ValueError: If ``values`` is empty or ``fraction`` is out of range.
    """
    if not values:
        raise ValueError("no samples")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def summarise(samples: list[float]) -> dict[str, float]:
    """Summarise per-item latencies given in seconds.

    Args:
        samples: Latency per item, in seconds.

    Returns:
        ``{"median_ms", "p95_ms", "mean_ms", "items_per_s"}``.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("no samples")
    ms = [value * 1000.0 for value in samples]
    mean_ms = statistics.fmean(ms)
    return {
        "median_ms": statistics.median(ms),
        "p95_ms": percentile(ms, 0.95),
        "mean_ms": mean_ms,
        "items_per_s": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


def nearest_rank(values: list[float], fraction: float) -> float:
    """Alias kept for readability at call sites that think in ranks."""
    return percentile(values, fraction)
