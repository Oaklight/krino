#!/usr/bin/env python3
"""Compare calibration artifacts, and refuse when they are not comparable.

A calibration number means nothing without the slice it was measured on. Three
times in one night this project compared figures that came from different slices
and drew the wrong conclusion each time: epochs 1 and 10 on a 111-item tier, an
unpaired pair of point estimates where a paired test was available, and a corpus
before and after #177 dropped the STS-B rows whose gold label was never observed.

The last of those is checkable from the artifact alone. `eval_calibration.py`
records the item count behind every level scale, so a repaired and an unrepaired
run of the same eval are distinguishable by the six-level `n` alone — 219 rows
apart on the 2026-10 corpus. This script prints the slices side by side and
withholds the deltas when they do not match, rather than printing a difference
that belongs to two different measurements.

Usage:
    python model/scripts/compare_calibrations.py \
        model/experiments/cal_xenc_split138_s123_ep3.json \
        model/experiments/cal_4b_split138_s123_ep5_repaired.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Two runs of the same eval draw the same number of items per type unless the
# corpus changed between them, so this is a slice check rather than a noise
# allowance. It is deliberately tighter than any sampling difference the eval
# observed in practice (the same split redrawn twice has matched to the item).
COUNT_TOLERANCE = 0.0


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "by_type" not in payload:
        raise SystemExit(f"{path} has no by_type block; is it a calibration artifact?")
    return payload


def type_counts(artifact: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(block.get("n", 0))
        for name, block in (artifact.get("by_type") or {}).items()
    }


def score_scales(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    score = (artifact.get("by_type") or {}).get("score") or {}
    return (score.get("score_metrics") or {}).get("qwk_by_levels") or {}


def ece_after_temperature(artifact: dict[str, Any]) -> dict[str, float]:
    """Cross-fitted holdout ECE, the number the clause is read from."""
    out: dict[str, float] = {}
    for name, block in (artifact.get("by_type") or {}).items():
        holdout = ((block.get("temperature") or {}).get("holdout") or {})
        if "ece_after" in holdout:
            out[name] = holdout["ece_after"]
    return out


def report(name: str, artifact: dict[str, Any]) -> None:
    counts = type_counts(artifact)
    ece = ece_after_temperature(artifact)
    print(f"\n== {name}")
    print(f"   checkpoint: {artifact.get('checkpoint', '(unrecorded)')}")
    print(f"   config:     {artifact.get('config', '(unrecorded)')}")
    print(f"   seed:       {artifact.get('seed', '(unrecorded)')}")
    print(f"   {'type':<8} {'n':>7} {'ECE@T':>8}")
    for q_type in sorted(counts):
        value = ece.get(q_type)
        shown = f"{value:.4f}" if value is not None else "—"
        print(f"   {q_type:<8} {counts[q_type]:>7} {shown:>8}")
    scales = score_scales(artifact)
    if scales:
        print(f"   {'scale':<8} {'n':>7} {'QWK':>8}")
        for level in sorted(scales, key=int):
            block = scales[level]
            print(f"   {level:<8} {int(block['n']):>7} {block['qwk']:>8.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="print deltas even when the slices differ (the numbers will not mean anything)",
    )
    args = parser.parse_args()
    if len(args.artifacts) < 2:
        parser.error("give at least two artifacts to compare")

    loaded = [(path.name, load(path)) for path in args.artifacts]
    for name, artifact in loaded:
        report(name, artifact)

    baseline_name, baseline = loaded[0]
    mismatches: list[str] = []
    for name, artifact in loaded[1:]:
        counts_a, counts_b = type_counts(baseline), type_counts(artifact)
        for q_type in sorted(set(counts_a) | set(counts_b)):
            a, b = counts_a.get(q_type, 0), counts_b.get(q_type, 0)
            if a != b:
                mismatches.append(f"{q_type}: {baseline_name} n={a} vs {name} n={b}")
        scales_a, scales_b = score_scales(baseline), score_scales(artifact)
        for level in sorted(set(scales_a) | set(scales_b), key=int):
            a = int((scales_a.get(level) or {}).get("n", 0))
            b = int((scales_b.get(level) or {}).get("n", 0))
            if a != b:
                mismatches.append(
                    f"score level {level}: {baseline_name} n={a} vs {name} n={b}"
                )

    if mismatches and not args.force:
        print("\nNOT COMPARABLE — the artifacts were measured on different slices:")
        for line in mismatches:
            print(f"  {line}")
        print(
            "\nA level scale's n differing is the signature of the STS-B repair "
            "(#177 drops rows whose gold label was never observed, 219 of them on "
            "the 2026-10 corpus), or of a different eval draw. Re-run one of the "
            "runs on the other's slice, or pass --force and do not quote the deltas."
        )
        return 1

    print("\nDeltas (later minus baseline):")
    for q_type in sorted(ece_after_temperature(baseline)):
        base = ece_after_temperature(baseline)[q_type]
        for name, artifact in loaded[1:]:
            other = ece_after_temperature(artifact).get(q_type)
            if other is not None:
                print(f"  ECE@T {q_type:<7} {other - base:+.4f}  ({name})")
    base_scales = score_scales(baseline)
    for level in sorted(base_scales, key=int):
        for name, artifact in loaded[1:]:
            other = (score_scales(artifact).get(level) or {}).get("qwk")
            if other is not None:
                delta = other - base_scales[level]["qwk"]
                print(f"  QWK level {level:<3} {delta:+.4f}  ({name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
