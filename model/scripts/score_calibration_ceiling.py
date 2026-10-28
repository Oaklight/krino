"""How far can a post-hoc confidence map move ECE, and where does it stop?

Reads a ``--predictions`` dump written by ``model/scripts/eval_calibration.py``
and, for each question type, reproduces the harness's own fit/holdout split --
a shuffle of ``range(n)`` keyed by ``"<seed>:<type>:temperature"``, which the
dump's item order pins down -- then compares three recalibrations on the same
holdout items:

* nothing (ECE at T=1),
* a monotone map fitted on the mirror half and evaluated on the holdout,
* the map's ECE again, with the map's regularisation chosen by an inner
  cross-validation on the fitting half rather than by the holdout.

The point of the third protocol is that the ``alpha``/``min_n`` grid must not be
selected on the data the result is reported on: choosing it there is worth about
0.005 of ECE on the score primitive, which is a quarter of the gap the map
closes.

Example:
    python model/scripts/score_calibration_ceiling.py \\
        --predictions model/experiments/cal_clean3_4b_predictions.jsonl \\
        --temperature-report model/experiments/cal_clean3_4b.json \\
        --json model/experiments/score_calibration_map_4b.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.evaluation.calibration import (  # noqa: E402
    apply_confidence_map,
    expected_calibration_error,
    fit_confidence_map,
)

ALPHAS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
MIN_NS = (1, 10, 20, 40, 80)


def harness_split(count: int, q_type: str, seed: int, fit_frac: float) -> tuple[list[int], list[int]]:
    """Reproduce ``_temperature_report``'s split over the dump's item order."""
    order = list(range(count))
    random.Random(f"{seed}:{q_type}:temperature").shuffle(order)
    n_fit = max(2, min(int(round(count * fit_frac)), count - 2))
    return order[:n_fit], order[n_fit:]


def ece(pairs: list[tuple[float, bool]]) -> float:
    return expected_calibration_error(
        [confidence for confidence, _ in pairs],
        [correct for _, correct in pairs],
    )["ece"]


def choose_regularisation(
    records: list[dict], fit: list[int], seed: str = "inner"
) -> tuple[tuple[float, int], dict]:
    """Pick ``(alpha, min_n)`` by two-fold cross-validation inside the fit half."""
    inner = list(fit)
    random.Random(seed).shuffle(inner)
    cut = len(inner) // 2
    halves = [(inner[:cut], inner[cut:]), (inner[cut:], inner[:cut])]
    scores: dict[tuple[float, int], float] = {}
    for alpha in ALPHAS:
        for min_n in MIN_NS:
            values = []
            for train, test in halves:
                mapping = fit_confidence_map(
                    [records[i]["top_prob"] for i in train],
                    [records[i]["correct"] for i in train],
                    alpha=alpha,
                    min_n=min_n,
                )
                mapped = apply_confidence_map(mapping, [records[i]["top_prob"] for i in test])
                values.append(ece([(m, bool(records[i]["correct"])) for m, i in zip(mapped, test)]))
            scores[(alpha, min_n)] = sum(values) / len(values)
    best = min(scores, key=scores.get)
    return best, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to a --predictions JSONL dump")
    parser.add_argument(
        "--temperature-report",
        default=None,
        help="Optional eval_calibration JSON; its holdout ECE after the fitted temperature is "
        "reported beside the map, on the same items",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-frac", type=float, default=0.5)
    parser.add_argument("--types", default="noul,choice,score")
    parser.add_argument("--json", default=None, help="Write the derived report here")
    args = parser.parse_args()

    records: dict[str, list[dict]] = defaultdict(list)
    with open(args.predictions, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[row.get("type")].append(row)

    temperature = {}
    if args.temperature_report:
        with open(args.temperature_report, encoding="utf-8") as handle:
            report = json.load(handle)
        temperature = {
            q_type: block.get("temperature", {}).get("holdout", {}).get("ece_after")
            for q_type, block in (report.get("by_type") or {}).items()
        }

    out: dict[str, object] = {"predictions": str(args.predictions), "by_type": {}}
    print("| type | holdout | ECE T=1 | after fitted T | after map (at a fixed 10/20) | after map (CV) | blocks |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for q_type in [token for token in args.types.split(",") if token.strip()]:
        rows = records.get(q_type) or []
        if len(rows) < 100:
            print(f"| {q_type} | {len(rows)} | too few items | | | | |")
            continue
        fit, hold = harness_split(len(rows), q_type, args.seed, args.fit_frac)
        holdout_pairs = [(rows[i]["top_prob"], bool(rows[i]["correct"])) for i in hold]
        before = ece(holdout_pairs)
        harness_before = None
        if args.temperature_report:
            harness_before = (report["by_type"][q_type]["temperature"]["holdout"].get("ece_before"))
        if harness_before is not None and abs(harness_before - before) > 5e-4:
            print(
                f"warning: {q_type} holdout ECE {before:.4f} does not reproduce the report's "
                f"{harness_before:.4f}; the dump and the report are not from the same pass",
                file=sys.stderr,
            )

        fixed = fit_confidence_map(
            [rows[i]["top_prob"] for i in fit],
            [rows[i]["correct"] for i in fit],
            alpha=10.0,
            min_n=20,
        )
        fixed_ece = ece(
            [
                (mapped, bool(rows[i]["correct"]))
                for mapped, i in zip(
                    apply_confidence_map(fixed, [rows[i]["top_prob"] for i in hold]), hold
                )
            ]
        )

        (alpha, min_n), scores = choose_regularisation(rows, fit)
        chosen = fit_confidence_map(
            [rows[i]["top_prob"] for i in fit],
            [rows[i]["correct"] for i in fit],
            alpha=alpha,
            min_n=min_n,
        )
        cv_ece = ece(
            [
                (mapped, bool(rows[i]["correct"]))
                for mapped, i in zip(
                    apply_confidence_map(chosen, [rows[i]["top_prob"] for i in hold]), hold
                )
            ]
        )

        after_t = temperature.get(q_type)
        print(
            f"| {q_type} | {len(hold)} | {before:.4f} | "
            f"{'-' if after_t is None else f'{after_t:.4f}'} | {fixed_ece:.4f} | "
            f"{cv_ece:.4f} | {len(chosen['blocks'])} |"
        )
        out["by_type"][q_type] = {
            "n_fit": len(fit),
            "n_holdout": len(hold),
            "ece_t1": before,
            "ece_after_temperature": after_t,
            "ece_after_map_fixed": fixed_ece,
            "ece_after_map_cv": cv_ece,
            "chosen": {"alpha": alpha, "min_n": min_n},
            "inner_cv": {f"{a}/{m}": v for (a, m), v in sorted(scores.items())},
            "map": chosen,
        }

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
