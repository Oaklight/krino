#!/usr/bin/env python3
"""Measure the calibration of the routed choice system, from dumped score vectors.

The rule that reaches the project's best overall accuracy sends short lists to
the backbone's native readout and long lists to the trained head, and its own
calibration has not been reported: the two paths calibrate separately, and a
pooled number is a different quantity from either. This script reads the
per-item records that ``model/scripts/blend_scorers.py --dump-records`` writes
and computes ECE and Brier for the head alone and for the routed system, with
one temperature per path fitted by cross-fitting.

No GPU is needed: the records carry both score vectors.

Usage:
    python model/scripts/eval_routed_calibration.py \
        --records s123_ep10=model/experiments/blend_records_full_s123_ep10.json \
        --records s123_ep5=model/experiments/blend_records_full_s123_ep5.json \
        --records s42_ep5=model/experiments/blend_records_full_s42_ep5.json \
        --folds 4 --json model/experiments/routed_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.routed_calibration import buckets, by_path, routed_report


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a ``--dump-records`` file, in either the wrapped or bare form."""
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("records", "items"):
            if key in payload:
                return list(payload[key])
        raise SystemExit(f"Error: {path} has no 'records' key")
    return list(payload)


def parse_name_path(value: str) -> tuple[str, Path]:
    """Split ``name=path``; a bare path is named by its stem."""
    if "=" in value:
        name, _, raw = value.partition("=")
        return name, Path(raw)
    return Path(value).stem, Path(value)


def print_report(name: str, report: dict[str, Any]) -> None:
    mean = report["mean"]
    print(f"\n{name}  paths={report['paths']}  folds={report['rule']['folds']}")
    print(f"  {'':24s} {'ECE':>8s} {'Brier':>8s} {'acc':>8s}")
    for metric, label in (
        ("head_only", "head only (T=1)"),
        ("head_only_calibrated", "head only (fitted T)"),
        ("routed_at_t1", "routed (T=1)"),
        ("routed_single_t", "routed (one T for everything)"),
        ("routed", "routed (fitted T per path)"),
    ):
        row = mean[metric]
        print(
            f"  {label:24s} {row['ece']:8.4f} {row['brier']:8.4f} {row['accuracy']:8.4f}"
        )
    temperatures = [fold["temperature"] for fold in report["folds"]]
    native = [value["native"] for value in temperatures]
    head = [value["head"] for value in temperatures]
    print(f"  fitted T per fold: native {native} head {head}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Per-item dump from blend_scorers.py; repeat for each checkpoint",
    )
    parser.add_argument("--folds", type=int, default=4, help="Cross-fitting folds")
    parser.add_argument(
        "--criterion",
        choices=["ece", "nll"],
        default="ece",
        help="What the temperature minimises (the goal names ECE)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the artifact here")
    args = parser.parse_args(argv)

    artifact: dict[str, Any] = {"folds": args.folds, "criterion": args.criterion, "runs": {}}
    for value in args.records:
        name, path = parse_name_path(value)
        records = load_records(path)
        report = routed_report(records, folds=args.folds, criterion=args.criterion)
        report["buckets"] = buckets(records)
        report["source"] = str(path)
        artifact["runs"][name] = report
        print_report(name, report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
