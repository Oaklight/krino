"""Torch-free aggregation of evaluation results into per-type and per-source views.

Lives outside ``model/training`` so the arithmetic can be unit-tested on the CI
interpreter, which installs only ``pytest`` — no torch. ``eval_epoch_detailed``
collects one row per scored item and hands it here.

The breakdown separates two questions a single flat table cannot answer:

- which question type is weak (``by_type``), and
- which data source is weak (``by_source_type``).

``by_source`` is kept for callers that compare single-type sources, but a source
that contributes more than one type has no single ``type``; it reports ``None``
rather than naming whichever type happened to arrive first.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# A cell is flagged in the printed table when its accuracy point estimate looks
# worse than this many standard errors below the panel's weighted mean. It is a
# display aid, not a statistical test.
_FLAG_Z = 1.0

# (source, question type, loss, is_correct) with two optional trailing columns,
# (rps, mae), carried only by score items -- the two metrics that respect the
# order of the levels. A row of length 4 says "not applicable".
Row = tuple[Any, ...]


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval ``(lower, upper)`` for a binomial proportion.

    Argument for the ordering below: the table exists to answer "which data do I
    fix next", so the row that belongs at the top is the one whose accuracy is
    *reliably* low — not the one we know least about.

    Sorting ascending by the lower bound does the wrong thing: 1/3 has a lower
    bound of 0.061 against 0.272 for 300/1000, so a three-item cell would lead
    the list purely by being thin. Sorting ascending by the **upper** bound puts
    the solidly-poor cell first (upper 0.331 for 300/1000), leaves the thin cell
    in the middle (upper 0.792 for 1/3), and puts a healthy cell last (upper
    0.952 for 90/100). The lower bound is still printed, so a reader can see how
    much of each point estimate is established.

    Args:
        correct: number of correct predictions.
        n: number of items; returns ``(0.0, 1.0)`` for n == 0, i.e. no
            information either way.
        z: two-sided multiplier; 1.96 is the 95% normal value.

    Returns:
        ``(lower, upper)`` in [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = correct / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (
        max(0.0, (centre - spread) / denom),
        min(1.0, (centre + spread) / denom),
    )


def wilson_lower_bound(correct: int, n: int, z: float = 1.96) -> float:
    """Lower endpoint of :func:`wilson_interval`."""
    return wilson_interval(correct, n, z)[0]


def wilson_upper_bound(correct: int, n: int, z: float = 1.96) -> float:
    """Upper endpoint of :func:`wilson_interval`; the ordering key."""
    return wilson_interval(correct, n, z)[1]


def _metrics(
    total_loss: float,
    n: int,
    correct: int,
    rps: list[float] | None = None,
    mae: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate one bucket.

    ``rps`` and ``mae`` are the per-item values of the score items in the
    bucket; when the bucket holds none, both keys are absent rather than zero,
    so a consumer can tell "no score items" from "perfect score".
    """
    metrics = {
        "mean_loss": total_loss / max(n, 1),
        "accuracy": correct / max(n, 1),
        "n_items": n,
        "correct": int(correct),
        "wilson_lower": round(wilson_lower_bound(correct, n), 4),
        "wilson_upper": round(wilson_upper_bound(correct, n), 4),
    }
    if rps:
        metrics["rps"] = sum(rps) / len(rps)
        metrics["mae"] = sum(mae) / len(mae)
        metrics["n_scored"] = len(rps)
    return metrics


def build_eval_breakdown(rows: Iterable[Row]) -> dict[str, Any]:
    """Aggregate ``(source, type, loss, is_correct)`` rows into eval statistics.

    Args:
        rows: one tuple per scored item, in any order.

    Returns:
        Dict with keys:
        - ``aggregate``: overall metrics
        - ``by_type``: ``{type: metrics}``
        - ``by_source``: ``{source: metrics + "type"}``, ``type`` is ``None``
          when the source contributes more than one question type
        - ``by_source_type``: ``{source: {type: metrics}}``
    """
    total_loss = 0.0
    n_items = 0
    correct = 0
    total_rps: list[float] = []
    total_mae: list[float] = []

    type_stats: dict[str, list] = {}
    source_stats: dict[str, list] = {}
    source_type_stats: dict[tuple[str, str], list] = {}
    type_scores: dict[str, tuple[list[float], list[float]]] = {}
    source_scores: dict[str, tuple[list[float], list[float]]] = {}
    source_type_scores: dict[tuple[str, str], tuple[list[float], list[float]]] = {}

    for row in rows:
        source, q_type, loss_val, is_correct = row[:4]
        # Score items carry (rps, mae); every other row stops at four columns.
        rps_val = row[4] if len(row) > 4 else None
        mae_val = row[5] if len(row) > 5 else None

        total_loss += loss_val
        n_items += 1
        if is_correct:
            correct += 1

        for stats, key in (
            (type_stats, q_type),
            (source_stats, source),
            (source_type_stats, (source, q_type)),
        ):
            bucket = stats.setdefault(key, [0.0, 0, 0])
            bucket[0] += loss_val
            bucket[1] += 1
            bucket[2] += int(is_correct)

        if rps_val is not None:
            total_rps.append(rps_val)
            total_mae.append(mae_val if mae_val is not None else float("nan"))
            for scores, key in (
                (type_scores, q_type),
                (source_scores, source),
                (source_type_scores, (source, q_type)),
            ):
                bucket_rps, bucket_mae = scores.setdefault(key, ([], []))
                bucket_rps.append(rps_val)
                bucket_mae.append(mae_val if mae_val is not None else float("nan"))

    by_type = {
        t: _metrics(*s, *type_scores.get(t, (None, None)))
        for t, s in sorted(type_stats.items())
    }

    by_source_type: dict[str, dict[str, dict[str, Any]]] = {}
    for (source, q_type), stats in sorted(source_type_stats.items()):
        by_source_type.setdefault(source, {})[q_type] = _metrics(
            *stats, *source_type_scores.get((source, q_type), (None, None))
        )

    by_source: dict[str, dict[str, Any]] = {}
    for source, stats in sorted(source_stats.items()):
        entry = _metrics(*stats, *source_scores.get(source, (None, None)))
        types = by_source_type.get(source, {})
        entry["type"] = next(iter(types)) if len(types) == 1 else None
        by_source[source] = entry

    return {
        "aggregate": _metrics(
            total_loss, n_items, correct, total_rps or None, total_mae or None
        ),
        "by_type": by_type,
        "by_source": by_source,
        "by_source_type": by_source_type,
    }


def format_detailed_eval(stats: dict[str, Any]) -> str:
    """Render an eval breakdown as a report.

    Source rows are ordered by Wilson lower bound, worst first, so a small cell
    with a lucky point estimate cannot lead the list. The fallback to
    ``by_source`` keeps pre-existing ``history.json`` files readable, which have
    no ``by_source_type``.

    Returns:
        The report as a single string, newline-terminated.
    """
    lines: list[str] = []
    agg = stats.get("aggregate", {})
    if agg:
        lines.append(
            f"  Aggregate: loss={agg['mean_loss']:.4f} acc={agg['accuracy']:.4f} "
            f"({agg['n_items']} items)"
        )

    by_type = stats.get("by_type", {})
    if by_type:
        lines.append("")
        lines.append("  By type:")
        for t in sorted(by_type):
            ts = by_type[t]
            extra = ""
            if "rps" in ts:
                # Only score buckets have them: the metrics that respect the
                # order of the levels.
                extra = f" rps={ts['rps']:.4f} mae={ts['mae']:.4f}"
            lines.append(
                f"    {t:<10} loss={ts['mean_loss']:.4f} acc={ts['accuracy']:.4f} "
                f"({ts['n_items']} items){extra}"
            )

    by_source_type = stats.get("by_source_type")
    rows: list[tuple[str, str, dict[str, Any]]] = []
    if by_source_type:
        rows = [
            (src, t, m)
            for src, types in by_source_type.items()
            for t, m in types.items()
        ]
    else:
        for src, m in stats.get("by_source", {}).items():
            rows.append((src, m.get("type") or "-", m))

    def _upper_bound(metrics: dict[str, Any]) -> float:
        """Wilson upper bound, computed on the fly for pre-existing history files."""
        if "wilson_upper" in metrics:
            return metrics["wilson_upper"]
        return wilson_upper_bound(metrics.get("correct", 0), metrics.get("n_items", 0))

    def _lower_bound(metrics: dict[str, Any]) -> float:
        if "wilson_lower" in metrics:
            return metrics["wilson_lower"]
        return wilson_lower_bound(metrics.get("correct", 0), metrics.get("n_items", 0))

    if rows:
        # Worst *reliably* first: a thin cell must not lead the list just because
        # its point estimate is volatile (see wilson_interval).
        rows.sort(key=lambda r: (_upper_bound(r[2]), -r[2]["n_items"], r[0], r[1]))

    if rows:
        lines.append("")
        lines.append(f"  {'Source':<20} {'Type':<8} {'Loss':>8} {'Acc':>8} {'N':>6} {'LB':>6}")
        lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6}")
        for src, t, m in rows:
            lb = _lower_bound(m)
            flag = ""
            if agg and m["n_items"] >= 20:
                se = math.sqrt(max(agg["accuracy"] * (1 - agg["accuracy"]), 1e-9) / m["n_items"])
                if m["accuracy"] < agg["accuracy"] - _FLAG_Z * se:
                    flag = "  <--"
            lines.append(
                f"  {src:<20} {t:<8} {m['mean_loss']:>8.4f} {m['accuracy']:>8.4f} "
                f"{m['n_items']:>6} {lb:>6.3f}{flag}"
            )

    return "\n".join(lines) + "\n"
