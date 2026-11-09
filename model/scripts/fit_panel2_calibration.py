#!/usr/bin/env python3
"""Fit per-type panel-2 calibration on dev and evaluate once on JevBench (#338).

Panel 2 (:mod:`model.evaluation.panel2`) gives K raw code logits per item.  This
script fits a per-type temperature and a per-type per-slot bias on the
development set, then applies the frozen parameters to the committed JevBench
predictions and reports accuracy, ECE, the multiclass Brier score and the
reliability table before and after.

The fit never sees JevBench.  The JevBench input is the committed panel-2 cell
from #326; calibration only transforms stored logits, so no second model
evaluation is needed and the acceptance instrument is not re-read.

Two facts shape the report, and both are checked rather than asserted:

* a temperature is monotone, so accuracy is bit-identical before and after;
* a bias equal on every slot cancels in the softmax, so the fitted "bias" is a
  per-slot vector, which is a position prior as much as a calibration term.

Usage:
    python model/scripts/fit_panel2_calibration.py \\
        --dev results/jevbench/panel2_dev_Qwen_Qwen3-Reranker-4B.json \\
        --jevbench results/jevbench/panel2_Qwen_Qwen3-Reranker-4B.json \\
        --output results/jevbench/panel2_calibration_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.calibration import (  # noqa: E402
    fit_temperature,
    fit_temperature_bias,
    multiclass_brier,
    reliability_diagram_data,
    softmax_logits,
    top1_calibration,
)

COMPATIBLE_FIELDS = ("prompt_template", "code_token_form", "readout", "strategy", "norm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows_from(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for row in payload["predictions_natural"]:
        rows.append(
            {
                "id": row["id"],
                "type": row["type"],
                "logits": [float(value) for value in row["code_logits"]],
                "gold": int(row["gold_pos"]),
                "source": source,
            }
        )
    return rows


def check_pair(dev: dict[str, Any], jevbench: dict[str, Any], dev_path: Path, jev_path: Path) -> None:
    left, right = dev["_provenance"], jevbench["_provenance"]
    if left["model_id"] != right["model_id"]:
        raise SystemExit(
            f"model mismatch: dev has {left['model_id']}, JevBench has {right['model_id']}"
        )
    for field in COMPATIBLE_FIELDS:
        if left.get(field) != right.get(field):
            raise SystemExit(
                f"{field} differs between {dev_path.name} and {jev_path.name}: "
                f"{left.get(field)!r} vs {right.get(field)!r}"
            )
    if dev["_provenance"].get("dataset") != "dev":
        raise SystemExit(f"{dev_path.name} is not a dev run")
    if jevbench["_provenance"].get("dataset", "jevbench") != "jevbench":
        raise SystemExit(f"{jev_path.name} is not a JevBench run")


def by_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {name: [row for row in rows if row["type"] == name] for name in sorted({row["type"] for row in rows})}


def probabilities(
    rows: list[dict[str, Any]],
    temperature: float,
    bias: list[float] | None = None,
) -> list[list[float]]:
    return [softmax_logits(row["logits"], temperature, bias) for row in rows]


def metrics(distributions: list[list[float]], labels: list[int], bins: int) -> dict[str, Any]:
    if not distributions:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0, "mean_confidence": 0.0}
    predictions = [max(range(len(p)), key=p.__getitem__) for p in distributions]
    confidences = [max(p) for p in distributions]
    return {
        "n": len(distributions),
        "accuracy": sum(int(p == label) for p, label in zip(predictions, labels)) / len(labels),
        "ece": top1_calibration(distributions, labels, bins)["ece"],
        "brier": multiclass_brier(distributions, labels)["brier"],
        "mean_confidence": sum(confidences) / len(confidences),
    }


def evaluate(
    rows: list[dict[str, Any]],
    fitted: dict[str, Any],
    mode: str,
    bins: int,
) -> dict[str, Any]:
    """Score ``rows`` under one parameterisation; ``mode`` names the family."""
    types = by_type(rows)
    per_type: dict[str, Any] = {}
    pooled_distributions: list[list[float]] = []
    pooled_labels: list[int] = []
    pooled_rows: list[dict[str, Any]] = []
    for name, subset in types.items():
        entry = fitted[name]
        if mode == "before":
            distribution = probabilities(subset, 1.0, None)
        elif mode == "temperature":
            distribution = probabilities(subset, entry["temperature"])
        elif mode == "temperature_bias":
            distribution = probabilities(
                subset, entry["temperature_bias"]["temperature"], entry["temperature_bias"]["bias"]
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        labels = [row["gold"] for row in subset]
        scored = metrics(distribution, labels, bins)
        scored["reliability"] = reliability_diagram_data(
            [max(p) for p in distribution],
            [
                max(range(len(p)), key=p.__getitem__) == label
                for p, label in zip(distribution, labels)
            ],
            bins,
        )
        per_type[name] = scored
        pooled_distributions.extend(distribution)
        pooled_labels.extend(labels)
        pooled_rows.extend(subset)
    overall = metrics(pooled_distributions, pooled_labels, bins)
    overall["reliability"] = reliability_diagram_data(
        [max(p) for p in pooled_distributions],
        [
            max(range(len(p)), key=p.__getitem__) == label
            for p, label in zip(pooled_distributions, pooled_labels)
        ],
        bins,
    )
    overall["predictions"] = [
        max(range(len(p)), key=p.__getitem__) for p in pooled_distributions
    ]
    overall["ids"] = [row["id"] for row in pooled_rows]
    return {"overall": overall, "by_type": per_type}


def fit_all(rows: list[dict[str, Any]], l2: float, bounds: tuple[float, float]) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for name, subset in by_type(rows).items():
        logits = [row["logits"] for row in subset]
        labels = [row["gold"] for row in subset]
        temperature = fit_temperature(logits, labels, criterion="nll")
        joint = fit_temperature_bias(logits, labels, l2=l2, bounds=bounds)
        fitted[name] = {
            "n": len(subset),
            "n_slots": max(len(row) for row in logits),
            "temperature": temperature["temperature"],
            "temperature_fit": temperature,
            "temperature_bias": joint,
        }
    return fitted


def mcnemar_change(
    before: dict[str, int], after: dict[str, int], gold: dict[str, int]
) -> dict[str, Any]:
    """Exact paired McNemar between the before and after predictions.

    The two evaluations pool their rows in type order, which is not the file
    order of the JevBench dump, so the comparison is made on item ids rather
    than on list position.  The counts are reconciled against the accuracy
    totals and the run aborts if they disagree, which is what a positional
    misalignment would look like.
    """
    from model.scripts.paired_mcnemar import mcnemar

    shared = sorted(set(before) & set(after) & set(gold))
    left = {item_id: before[item_id] == gold[item_id] for item_id in shared}
    right = {item_id: after[item_id] == gold[item_id] for item_id in shared}
    wins, losses, p_value = mcnemar(right, left, shared)
    before_correct = sum(left.values())
    after_correct = sum(right.values())
    if after_correct - before_correct != wins - losses:
        raise SystemExit(
            "McNemar counts do not reconcile with the accuracy totals "
            f"({wins} - {losses} against {after_correct} - {before_correct})"
        )
    return {
        "n": len(shared),
        "after_only_correct": wins,
        "before_only_correct": losses,
        "net": wins - losses,
        "before_correct": before_correct,
        "after_correct": after_correct,
        "exact_mcnemar_p": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--jevbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--bias-bound", type=float, default=6.0)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument(
        "--fit-code-commit",
        default=None,
        help="commit of the fitting script, when it is not the commit that "
        "produced the evaluated predictions",
    )
    args = parser.parse_args()

    dev_payload = load_payload(args.dev)
    jev_payload = load_payload(args.jevbench)
    check_pair(dev_payload, jev_payload, args.dev, args.jevbench)

    dev_rows = rows_from(dev_payload, "dev")
    jev_rows = rows_from(jev_payload, "jevbench")
    fitted = fit_all(dev_rows, args.l2, (-args.bias_bound, args.bias_bound))

    before = evaluate(jev_rows, fitted, "before", args.bins)
    after_temperature = evaluate(jev_rows, fitted, "temperature", args.bins)
    after_bias = evaluate(jev_rows, fitted, "temperature_bias", args.bins)

    unchanged = before["overall"]["predictions"] == after_temperature["overall"]["predictions"]
    if not unchanged:
        raise SystemExit(
            "temperature scaling changed the argmax; that cannot happen and the "
            "logits or the parameters are wrong"
        )

    gold = {row["id"]: row["gold"] for row in jev_rows}

    def by_id(result: dict[str, Any]) -> dict[str, int]:
        return dict(zip(result["overall"]["ids"], result["overall"]["predictions"]))

    change = mcnemar_change(by_id(before), by_id(after_bias), gold)

    payload = {
        "_provenance": {
            "issue": "#338",
            "model_id": jev_payload["_provenance"]["model_id"],
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit": args.code_commit,
            "fit_code_commit": args.fit_code_commit or args.code_commit,
            "fit_dataset": "dev",
            "dev_file": args.dev.name,
            "dev_sha256": sha256(args.dev),
            "dev_items_scored": dev_payload["_provenance"].get("items"),
            "jevbench_file": args.jevbench.name,
            "jevbench_sha256": sha256(args.jevbench),
            "jevbench_items": len(jev_rows),
            "fit": {
                "criterion": "nll",
                "model": "p = softmax(z / T + b); T and b fitted per type on dev",
                "l2": args.l2,
                "bias_bounds": [-args.bias_bound, args.bias_bound],
                "bins": args.bins,
            },
            "temperature_is_monotone": True,
            "argmax_unchanged_under_temperature": unchanged,
        },
        "fit": fitted,
        "jevbench": {
            "before": before,
            "after_temperature": after_temperature,
            "after_temperature_bias": after_bias,
            "accuracy_change_temperature_bias": change,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def line(label: str, entry: dict[str, Any]) -> str:
        return (
            f"  {label:<20} acc {entry['accuracy']:.4f}  ece {entry['ece']:.4f}  "
            f"brier {entry['brier']:.4f}  conf {entry['mean_confidence']:.4f}"
        )

    print(f"{jev_payload['_provenance']['model_id']}  JevBench n={len(jev_rows)}")
    for scope in ("overall",):
        print(scope)
        print(line("before", before[scope]))
        print(line("temperature", after_temperature[scope]))
        print(line("temperature+bias", after_bias[scope]))
    for name in before["by_type"]:
        print(name)
        print(line("before", before["by_type"][name]))
        print(line("temperature", after_temperature["by_type"][name]))
        print(line("temperature+bias", after_bias["by_type"][name]))
    for name, entry in fitted.items():
        print(
            f"fit {name:<7} n={entry['n']:<4} T={entry['temperature']:.4f} "
            f"Tb={entry['temperature_bias']['temperature']:.4f} "
            f"sweeps={entry['temperature_bias']['sweeps']}"
        )
    print(f"accuracy change under bias: {change}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
