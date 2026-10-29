#!/usr/bin/env python3
"""Score every ordinal decoder on a dumped set of score-head logits.

The harness turns a score head's distribution into a level with the argmax, which
is the mode and is only optimal when every mistake costs the same. QWK weights a
miss by the square of its distance, so its Bayes-optimal decision is the
posterior mean. This measures the difference on real logits and checks it out of
sample, because a decoder chosen on the same items it is scored on proves
nothing.

Input is the dump ``model/scripts/score_vote_adjust.py --dump-logits`` writes:
one record per score item with its level count, its gold level and its logits.

Usage:
    python model/scripts/eval_score_decoders.py \
        --logits model/experiments/score_logits_s123_ep10.json \
        --json model/experiments/score_decoders_s123_ep10.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.calibration import quadratic_weighted_kappa  # noqa: E402
from model.evaluation.ordinal_decode import DECODERS  # noqa: E402
from model.evaluation.prior_adjustment import MIN_SCALE_ITEMS  # noqa: E402


def per_scale_metrics(
    rows: Sequence[dict[str, Any]], decoder: str
) -> dict[str, Any]:
    """QWK and accuracy per level scale under one decoder."""
    by_scale: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scale[row["k"]].append(row)

    metrics: dict[str, Any] = {}
    for n_levels, scale_rows in sorted(by_scale.items()):
        decode = DECODERS[decoder]
        predictions = [decode(row["logits"]) for row in scale_rows]
        golds = [row["gold"] for row in scale_rows]
        metrics[str(n_levels)] = {
            "n": len(scale_rows),
            "qwk": quadratic_weighted_kappa(predictions, golds, n_levels),
            "accuracy": sum(1 for p, g in zip(predictions, golds) if p == g) / len(scale_rows),
        }
    readable = {
        scale: value for scale, value in metrics.items() if value["n"] >= MIN_SCALE_ITEMS
    }
    readable_n = sum(value["n"] for value in readable.values())
    return {
        "by_scale": metrics,
        "mixed_scale_qwk": (
            sum(value["qwk"] * value["n"] for value in readable.values()) / readable_n
            if readable_n
            else 0.0
        ),
        "excluded_scales": sorted(set(metrics) - set(readable)),
    }


def cross_fitted(rows: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Choose the decoder on one half and score it on the other, both ways.

    The decoders are compared per level scale, because QWK lives within a scale
    and one decoder can win at five levels while losing at four.
    """
    rng = random.Random(seed)
    by_scale: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scale[row["k"]].append(row)

    result: dict[str, Any] = {}
    for n_levels, scale_rows in sorted(by_scale.items()):
        if len(scale_rows) < MIN_SCALE_ITEMS:
            continue
        indices = list(range(len(scale_rows)))
        rng.shuffle(indices)
        halves = {
            "a": [scale_rows[i] for i in indices[::2]],
            "b": [scale_rows[i] for i in indices[1::2]],
        }

        def qwk_on(subset: Sequence[dict[str, Any]], decoder: str) -> float:
            decode = DECODERS[decoder]
            return quadratic_weighted_kappa(
                [decode(row["logits"]) for row in subset],
                [row["gold"] for row in subset],
                n_levels,
            )

        chosen = {
            half: max(DECODERS, key=lambda name: qwk_on(halves[half], name))
            for half in ("a", "b")
        }
        result[str(n_levels)] = {
            "choose_on_a": chosen["a"],
            "applied_to_b": qwk_on(halves["b"], chosen["a"]),
            "choose_on_b": chosen["b"],
            "applied_to_a": qwk_on(halves["a"], chosen["b"]),
            "argmax_on_a": qwk_on(halves["a"], "argmax"),
            "argmax_on_b": qwk_on(halves["b"], "argmax"),
        }
    return result


def bootstrap_delta(
    rows: Sequence[dict[str, Any]],
    n_levels: int,
    better: str,
    baseline: str,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap of one decoder's QWK minus another's, within a scale."""
    scale_rows = [row for row in rows if row["k"] == n_levels]
    if len(scale_rows) < 2:
        return {}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [scale_rows[rng.randrange(len(scale_rows))] for _ in range(len(scale_rows))]
        golds = [row["gold"] for row in sample]
        good = quadratic_weighted_kappa(
            [DECODERS[better](row["logits"]) for row in sample], golds, n_levels
        )
        base = quadratic_weighted_kappa(
            [DECODERS[baseline](row["logits"]) for row in sample], golds, n_levels
        )
        deltas.append(good - base)
    deltas.sort()
    return {
        "median": statistics.median(deltas),
        "ci_low": deltas[int(0.025 * resamples)],
        "ci_high": deltas[int(0.975 * resamples) - 1],
        "share_positive": sum(1 for value in deltas if value > 0) / resamples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.logits.read_text(encoding="utf-8"))["items"]
    print(f"{len(rows)} score items", flush=True)

    report: dict[str, Any] = {"decoders": {}, "cross_fitted": {}, "bootstrap": {}}
    for decoder in DECODERS:
        report["decoders"][decoder] = per_scale_metrics(rows, decoder)
        summary = report["decoders"][decoder]
        per_scale = " ".join(
            f"k{scale}={value['qwk']:.4f}"
            for scale, value in summary["by_scale"].items()
            if value["n"] >= MIN_SCALE_ITEMS
        )
        print(
            f"  {decoder:<16} mixed QWK={summary['mixed_scale_qwk']:.4f}  {per_scale}",
            flush=True,
        )

    report["cross_fitted"] = cross_fitted(rows, seed=args.seed)
    print("\ncross-fitted decoder choice:", flush=True)
    for scale, values in report["cross_fitted"].items():
        print(
            f"  k={scale}: choose on a -> {values['choose_on_a']} -> b {values['applied_to_b']:.4f} "
            f"(argmax {values['argmax_on_b']:.4f}); choose on b -> {values['choose_on_b']} "
            f"-> a {values['applied_to_a']:.4f} (argmax {values['argmax_on_a']:.4f})",
            flush=True,
        )

    for scale in sorted({row["k"] for row in rows}):
        if sum(1 for row in rows if row["k"] == scale) < MIN_SCALE_ITEMS:
            continue
        report["bootstrap"][str(scale)] = bootstrap_delta(
            rows, scale, "expected_level", "argmax", args.resamples, args.seed
        )
        values = report["bootstrap"][str(scale)]
        print(
            f"  k={scale} expected - argmax: {values['median']:+.4f} "
            f"[{values['ci_low']:+.4f}, {values['ci_high']:+.4f}] "
            f"P(>0)={values['share_positive']:.3f}",
            flush=True,
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
