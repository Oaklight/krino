#!/usr/bin/env python3
"""Compare panel 2 with the judgement readout on the internal dev set (#326).

One checkpoint, one set of rows, two readouts: the panel-2 dev cell written by
``eval_jevbench_panel2.py --dataset dev`` and the committed judgement dev cell
(``dev_judgement_<model>.json``).  The output is the paired comparison the site
claim needs: exact McNemar, discordant counts both ways, the raw point estimate
against dev's detectable difference, and the same split by question type.

The two instruments are not pooled.  This script compares them row by row on
the rows both scored; it does not combine JevBench and dev into one number.

Usage:
    python model/scripts/compare_dev_panel2.py \\
        --panel2 results/jevbench/dev_panel2_Qwen_Qwen3-Reranker-4B.json \\
        --judgement model/experiments/logit_prov/dev_judgement_Qwen_Qwen3-Reranker-4B.json \\
        --output results/jevbench/dev_panel2_vs_judgement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.paired_power import detectable_delta
from model.scripts.paired_mcnemar import mcnemar


def _correct(predictions: list[dict[str, Any]]) -> dict[str, bool]:
    return {row["id"]: bool(row["correct"]) for row in predictions}


def _paired(first: dict[str, bool], second: dict[str, bool]) -> dict[str, Any]:
    shared = sorted(set(first) & set(second))
    wins, losses, p_value = mcnemar(first, second, shared)
    return {
        "n": len(shared),
        "panel2_only_correct": wins,
        "judgement_only_correct": losses,
        "net": wins - losses,
        "mcnemar_p": p_value,
    }


def compare(
    panel2_predictions: list[dict[str, Any]],
    judgement_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    panel2 = _correct(panel2_predictions)
    judgement = _correct(judgement_predictions)
    shared = sorted(set(panel2) & set(judgement))
    overall = _paired(panel2, judgement)

    panel2_correct = sum(panel2[i] for i in shared)
    judgement_correct = sum(judgement[i] for i in shared)
    delta_pp = (panel2_correct - judgement_correct) / len(shared) * 100.0
    discordance = (
        overall["panel2_only_correct"] + overall["judgement_only_correct"]
    ) / len(shared)
    detectable_pp = detectable_delta(len(shared), discordance) * 100.0

    by_type: dict[str, Any] = {}
    types = {row["id"]: row.get("type", "noul") for row in panel2_predictions}
    for q_type in sorted({types[i] for i in shared}):
        ids = [i for i in shared if types[i] == q_type]
        cell = _paired(
            {i: panel2[i] for i in ids}, {i: judgement[i] for i in ids}
        )
        cell["panel2_correct"] = sum(panel2[i] for i in ids)
        cell["judgement_correct"] = sum(judgement[i] for i in ids)
        cell["panel2_accuracy"] = cell["panel2_correct"] / len(ids)
        cell["judgement_accuracy"] = cell["judgement_correct"] / len(ids)
        cell["delta_pp"] = (cell["panel2_correct"] - cell["judgement_correct"]) / len(ids) * 100.0
        by_type[q_type] = cell

    return {
        "n": len(shared),
        "panel2_correct": panel2_correct,
        "judgement_correct": judgement_correct,
        "panel2_accuracy": panel2_correct / len(shared),
        "judgement_accuracy": judgement_correct / len(shared),
        "delta_pp": delta_pp,
        "discordance": discordance,
        "detectable_pp": detectable_pp,
        "overall": overall,
        "by_type": by_type,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel2", type=Path, required=True)
    parser.add_argument("--judgement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel2_payload = json.loads(args.panel2.read_text())
    judgement_payload = json.loads(args.judgement.read_text())
    result = compare(
        panel2_payload["predictions_natural"], judgement_payload["predictions"]
    )
    result["_provenance"] = {
        "panel2": {
            "path": _relative(args.panel2),
            "validity": panel2_payload["validity"],
            "code_commit": panel2_payload["_provenance"]["code_commit"],
            "config_sha256": panel2_payload["_provenance"].get("config_sha256"),
            "data_sha256": panel2_payload["_provenance"].get("data_sha256"),
        },
        "judgement": {
            "path": _relative(args.judgement),
            "code_commit": judgement_payload["_provenance"]["code_commit"],
        },
        "dataset": "dev (internal)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"panel2 {result['panel2_correct']}/{result['n']} vs judgement "
        f"{result['judgement_correct']}/{result['n']}, delta {result['delta_pp']:+.2f} pp, "
        f"+{result['overall']['panel2_only_correct']}/-"
        f"{result['overall']['judgement_only_correct']}, "
        f"p={result['overall']['mcnemar_p']:.4g}"
    )
    for q_type, cell in result["by_type"].items():
        print(
            f"  {q_type:<7} panel2 {cell['panel2_correct']}/{cell['n']} vs judgement "
            f"{cell['judgement_correct']}/{cell['n']}, delta {cell['delta_pp']:+.2f} pp, "
            f"p={cell['mcnemar_p']:.4g}"
        )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
