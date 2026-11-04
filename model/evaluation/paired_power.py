"""Exact power for a paired McNemar comparison on a fixed item set.

Every held-out comparison in this project ends up as the same question: two
checkpoints disagree on some of the same items, and we ask whether the split of
those disagreements is lopsided enough to believe.  On a fixed item set the
answer is already computable before any model runs, from the item count and the
rate at which the two models disagree.

That matters because the tier the goal names is small.  JevBench-public's hard
tier is 111 items, and the measured disagreement rate between two similar
checkpoints is about a third of them.  With 36 discordant items, an exact
two-sided McNemar test at 5% needs 26 or more of them to go one way before it
rejects -- a split that a genuine ten-point accuracy gain does not produce.  So
"the difference is not significant" on that tier is a statement about the tier,
not about the models, and it is worth being able to say by how much.

The model is deliberately minimal.  Let ``n`` be the items, ``r`` the fraction on
which the two systems disagree, and ``d`` the true accuracy difference.  The
number of disagreements that favour the first system is then binomial with
``n * r`` trials and success probability ``0.5 + d / (2 * r)``: the accuracy
difference is ``(b - c) / n`` and ``b + c = n * r``, so ``b = n * r * p`` gives
``p = 0.5 + d * n / (2 * n * r)``.  The test is the exact two-sided binomial one
used by :mod:`model.scripts.paired_mcnemar`, not the chi-square approximation,
because at 36 discordant pairs the approximation is visibly anti-conservative.

Nothing here imports outside the standard library, so it runs where torch does
not.
"""

from __future__ import annotations

from math import exp, lgamma, log


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    """Log of ``C(n, k) p**k (1 - p)**(n - k)``, computed without underflow.

    ``C(n, k)`` overflows the float range well before the discordant counts a
    large held-out set would produce, and ``(1 - p) ** n`` underflows to zero,
    so the mass is assembled in log space and exponentiated only at the end.
    """
    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log(1.0 - p)
    )


def two_sided_p(discordant_first: int, discordant_second: int) -> float:
    """Exact two-sided McNemar p-value for a split of the discordant pairs.

    Under the null the two directions are equally likely, so the count in the
    smaller direction is binomial with success probability one half.  The
    p-value doubles the smaller tail and caps at one.

    Args:
        discordant_first: Items the first system gets right and the second wrong.
        discordant_second: Items the second system gets right and the first wrong.

    Returns:
        The two-sided p-value in ``[0, 1]``.

    Raises:
        ValueError: If either count is negative.
    """
    if discordant_first < 0 or discordant_second < 0:
        raise ValueError(
            "discordant counts must be non-negative, got "
            f"{discordant_first} and {discordant_second}"
        )
    n = discordant_first + discordant_second
    if n == 0:
        return 1.0
    tail = min(discordant_first, discordant_second)
    mass = sum(exp(_log_binom_pmf(n, k, 0.5)) for k in range(tail + 1))
    return min(1.0, 2 * mass)


def _critical_index(n: int, alpha: float) -> int:
    """Largest ``k`` whose smaller tail still rejects at ``alpha``.

    The rejection region is ``min(b, n - b) <= k``; this returns the largest
    ``k`` that satisfies it, or ``-1`` when no split rejects.  The null masses
    are a binomial with success probability one half.
    """
    best = -1
    cumulative = 0.0
    for k in range(0, n // 2 + 1):
        cumulative += exp(_log_binom_pmf(n, k, 0.5))
        if min(1.0, 2.0 * cumulative) <= alpha:
            best = k
        else:
            break
    return best


def power(
    n_items: int,
    discordance_rate: float,
    delta: float,
    alpha: float = 0.05,
) -> float:
    """Probability that a paired McNemar test at ``n_items`` rejects.

    Args:
        n_items: Number of paired items.
        discordance_rate: Fraction of items on which the two systems disagree,
            measured on a comparable pair. Must be in ``(0, 1]``.
        delta: True accuracy difference between the two systems, as a fraction
            (``0.10`` for ten points). The sign does not matter: the test is
            two-sided.
        alpha: Significance level.

    Returns:
        The exact power in ``[0, 1]``.

    Raises:
        ValueError: If ``n_items`` is not positive, the rate is outside
            ``(0, 1]``, or ``alpha`` is outside ``(0, 1)``.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError(
            f"discordance_rate must be in (0, 1], got {discordance_rate}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n_discordant = max(1, round(discordance_rate * n_items))
    if n_discordant > n_items:
        n_discordant = n_items
    effect = abs(delta)
    p_win = 0.5 + effect * n_items / (2 * n_discordant)
    p_win = min(1.0, max(0.0, p_win))
    k_crit = _critical_index(n_discordant, alpha)
    if k_crit < 0:
        return 0.0
    if p_win <= 0.0:
        return 1.0
    if p_win >= 1.0:
        return 1.0 if n_discordant - k_crit <= n_discordant else 0.0
    total = sum(
        exp(_log_binom_pmf(n_discordant, wins, p_win))
        for wins in range(0, k_crit + 1)
    )
    total += sum(
        exp(_log_binom_pmf(n_discordant, wins, p_win))
        for wins in range(n_discordant - k_crit, n_discordant + 1)
    )
    return min(1.0, total)


def items_for_power(
    discordance_rate: float,
    delta: float,
    target: float = 0.8,
    alpha: float = 0.05,
    max_items: int = 100_000,
) -> int | None:
    """Smallest item count that reaches ``target`` power at a fixed outcome size.

    The discordance *rate* is held constant, so the number of discordant pairs
    grows with the item count. That is the right model when planning a larger
    held-out set drawn from the same distribution. Power is monotone in the
    item count apart from the integer rounding of the discordant count, so the
    boundary is found by bisection and then confirmed on the items just below
    it.

    Args:
        discordance_rate: Fraction of items the two systems disagree on.
        delta: True accuracy difference as a fraction.
        target: Desired power.
        alpha: Significance level.
        max_items: Give up past this count.

    Returns:
        The smallest ``n`` with power at least ``target``, or ``None`` if none
        up to ``max_items`` reaches it.

    Raises:
        ValueError: If ``target`` is outside ``(0, 1)`` or ``delta`` is zero.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target}")
    if delta == 0:
        raise ValueError("delta must be non-zero")
    lo, hi = 2, max_items
    if power(hi, discordance_rate, delta, alpha) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power(mid, discordance_rate, delta, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    # Rounding of the discordant count makes power dip slightly when the count
    # steps up, so bisection can land a few items high. Scan back over a window
    # wide enough to cover those dips and return the lowest count that clears
    # the target.
    for candidate in range(max(2, lo - 64), lo + 1):
        if power(candidate, discordance_rate, delta, alpha) >= target:
            return candidate
    return lo


def detectable_delta(
    n_items: int,
    discordance_rate: float,
    target: float = 0.8,
    alpha: float = 0.05,
    step: float = 0.001,
) -> float:
    """Smallest accuracy difference this item count detects at ``target`` power.

    Args:
        n_items: Number of paired items.
        discordance_rate: Fraction of items the two systems disagree on.
        target: Desired power.
        alpha: Significance level.
        step: Grid resolution for the search.

    Returns:
        The smallest ``delta`` in ``[0, 1]`` with power at least ``target``, or
        ``1.0`` if even a one-sided certainty at this size falls short.

    Raises:
        ValueError: If ``target`` is outside ``(0, 1)``.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target}")
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if power(n_items, discordance_rate, mid, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return round(hi / step) * step
