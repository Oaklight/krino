#!/usr/bin/env python3
"""Assemble the readout-by-capacity matrix on JevBench public (#329).

Two families, three checkpoints and three readouts each, on the same 231 items:

- reranker: Qwen3-Reranker-0.6B / 4B / 8B; the largest text reranker Qwen
  publishes, so capacity stops at 8B.
- general: Qwen3-4B / 14B / 32B (post-trained); the family that can keep growing,
  which is how the plateau is separated from the family.

Readouts: option-text continuation, per-option judgement, and panel 2.  The
paired comparisons are exact McNemar tests on the items each pair shares.  The
panel-2 cells pass through the #326 validity gate: a cell whose codes are not
all single tokens, or which has unscorable items, is marked invalid and is left
out of the matrix and every comparison.

Usage:
    python model/scripts/summarize_readout_capacity.py \\
        --output results/jevbench/readout_capacity_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.scripts.paired_mcnemar import mcnemar

FAMILIES: dict[str, tuple[str, ...]] = {
    "reranker": (
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen3-Reranker-8B",
    ),
    "general": ("Qwen/Qwen3-4B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"),
}
READOUTS = ("continuation", "judgement", "panel2")

RESULTS = ROOT / "results/jevbench"
LOGIT_PROV = ROOT / "model/experiments/logit_prov"


def cell_path(model: str, readout: str) -> Path:
    """Where the cell lives: new #329 files first, then the committed cells."""
    tag = model.replace("/", "_")
    if readout == "panel2":
        candidates = [RESULTS / f"panel2_{tag}.json"]
    elif readout == "continuation":
        candidates = [RESULTS / f"continuation_{tag}.json"]
    else:
        candidates = [
            RESULTS / f"judgement_{tag}.json",
            LOGIT_PROV / f"jevbench_judgement_{tag}.json",
        ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no file for {model} / {readout}: {candidates}")


def load_cell(model: str, readout: str) -> dict[str, Any]:
    """One cell's accuracy, per-item outcomes, latency and diagnostics."""
    path = cell_path(model, readout)
    payload = json.loads(path.read_text())

    if readout == "panel2":
        summary = payload["metrics"]["natural"]["summary"]
        predictions = {
            row["id"]: bool(row["correct"]) for row in payload["predictions_natural"]
        }
        latency = payload["metrics"]["latency"]["natural"]["mean_s"]
        validity = payload["validity"]
        order = payload["metrics"]["order_effect"]
        position = payload["metrics"]["natural"]["position"]
        code_commit = payload["_provenance"]["code_commit"]
    else:
        summary = payload["summary"]
        predictions = {
            row["id"]: bool(row["correct"]) for row in payload["predictions"]
        }
        latency = (payload.get("latency") or {}).get("mean_s")
        if latency is None:
            elapsed = payload.get("_provenance", {}).get("elapsed_s")
            total = summary["overall"]["total"]
            latency = elapsed / total if elapsed and total else None
        validity = {"valid": True, "single_token_codes": None, "unscorable_items": 0}
        order = None
        position = None
        code_commit = payload["_provenance"]["code_commit"]

    return {
        "readout": readout,
        "path": str(path.relative_to(ROOT)),
        "correct": summary["overall"]["correct"],
        "total": summary["overall"]["total"],
        "accuracy": summary["overall"]["accuracy"],
        "latency_mean_s": latency,
        "by_tier": summary.get("by_tier", {}),
        "by_type": summary.get("by_type", {}),
        "validity": validity,
        "order_effect": order,
        "position": position,
        "code_commit": code_commit,
        "predictions": predictions,
    }


def pair(first: dict[str, bool], second: dict[str, bool]) -> dict[str, Any]:
    """Exact two-sided McNemar on the shared items; first is the arm."""
    shared = sorted(set(first) & set(second))
    wins, losses, p_value = mcnemar(first, second, shared)
    return {
        "n": len(shared),
        "first_only_correct": wins,
        "second_only_correct": losses,
        "net": wins - losses,
        "mcnemar_p": p_value,
    }


def family_matrix(models: tuple[str, ...]) -> tuple[dict, dict, dict, dict]:
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    matrix: dict[str, Any] = {}
    predictions: dict[str, dict[str, bool]] = {}
    for model in models:
        cells[model] = {}
        matrix[model] = {}
        for readout in READOUTS:
            cell = load_cell(model, readout)
            cells[model][readout] = cell
            valid = bool(cell["validity"].get("valid", True))
            matrix[model][readout] = {
                "valid": valid,
                "correct": cell["correct"],
                "total": cell["total"],
                "accuracy": cell["accuracy"],
                "latency_mean_s": cell["latency_mean_s"],
                "by_tier": cell["by_tier"],
                "by_type": cell["by_type"],
                "path": cell["path"],
            }
            if valid:
                predictions[f"{model}|{readout}"] = cell["predictions"]

    short = {model: model.split("/")[-1] for model in models}

    capacity: dict[str, Any] = {}
    for readout in READOUTS:
        pairs = list(zip(models, models[1:])) + [(models[0], models[-1])]
        for low, high in pairs:
            first = predictions.get(f"{low}|{readout}")
            second = predictions.get(f"{high}|{readout}")
            if first is None or second is None:
                continue
            capacity[f"{readout}:{short[low]} vs {short[high]}"] = pair(first, second)

    readouts: dict[str, Any] = {}
    for model in models:
        for first_readout, second_readout in (
            ("judgement", "continuation"),
            ("panel2", "judgement"),
            ("panel2", "continuation"),
        ):
            first = predictions.get(f"{model}|{first_readout}")
            second = predictions.get(f"{model}|{second_readout}")
            if first is None or second is None:
                continue
            readouts[
                f"{short[model]}: {first_readout} vs {second_readout}"
            ] = pair(first, second)

    return cells, matrix, capacity, readouts


def build() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issue": 329,
        "items": 231,
        "families": {name: list(models) for name, models in FAMILIES.items()},
        "readouts": list(READOUTS),
        "matrix": {},
        "paired": {},
        "validity_gate": {},
        "order_effect": {},
        "provenance": {},
    }
    for family, models in FAMILIES.items():
        cells, matrix, capacity, readouts = family_matrix(models)
        payload["matrix"][family] = matrix
        payload["paired"][family] = {"capacity": capacity, "readout": readouts}
        payload["validity_gate"][family] = {
            model: {
                "valid": cells[model]["panel2"]["validity"].get("valid"),
                "single_token_codes": cells[model]["panel2"]["validity"].get(
                    "single_token_codes"
                ),
                "unscorable_items": cells[model]["panel2"]["validity"].get(
                    "unscorable_items"
                ),
            }
            for model in models
        }
        payload["order_effect"][family] = {
            model: cells[model]["panel2"]["order_effect"] for model in models
        }
        payload["provenance"][family] = {
            model: {
                readout: {
                    "path": cells[model][readout]["path"],
                    "code_commit": cells[model][readout]["code_commit"],
                }
                for readout in READOUTS
            }
            for model in models
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/jevbench/readout_capacity_matrix.json",
    )
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for family, matrix in payload["matrix"].items():
        for model, row in matrix.items():
            line = [model.split("/")[-1]]
            for readout in READOUTS:
                cell = row[readout]
                line.append(
                    f"{readout} {cell['correct']}/{cell['total']}"
                    if cell["valid"]
                    else f"{readout} INVALID"
                )
            print("  ".join(line))
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
