"""Risk-coverage evaluation for the abstention head.

Consumes a predictions JSONL with one record per item:

    {"id", "q_type", "n_options", "gold", "pred", "correct",
     "max_prob", "p_answerable", "reason"}

``p_answerable`` is the model's probability that the item is answerable;
``max_prob`` is the baseline confidence (the largest option probability).
Abstaining earns zero credit: utility is correct answers divided by all items,
while answered accuracy is reported separately.

Usage:
    python model/scripts/eval_abstention.py \
        --predictions predictions.jsonl \
        --out abstention_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _correct(record: dict) -> bool:
    if "correct" in record:
        return bool(record["correct"])
    return str(record.get("pred")) == str(record.get("gold"))


def _aurc(records: list[dict], score_field: str) -> float | None:
    """Area under the risk-coverage curve; lower is better."""
    scored = [record for record in records if record.get(score_field) is not None]
    if not scored:
        return None
    scored.sort(key=lambda record: (-float(record[score_field]), str(record.get("id"))))
    correct = 0
    total_risk = 0.0
    for index, record in enumerate(scored, 1):
        correct += int(_correct(record))
        total_risk += 1.0 - correct / index
    return total_risk / len(scored)


def _at_coverage(records: list[dict], score_field: str, coverage: float) -> dict:
    scored = [record for record in records if record.get(score_field) is not None]
    if not scored:
        return {"coverage": coverage, "answered": 0, "answered_accuracy": None,
                "abstain_zero_accuracy": None}
    scored.sort(key=lambda record: (-float(record[score_field]), str(record.get("id"))))
    answered = max(1, math.ceil(len(scored) * coverage / 100.0))
    subset = scored[:answered]
    correct = sum(int(_correct(record)) for record in subset)
    return {
        "coverage": coverage,
        "answered": answered,
        "answered_accuracy": correct / answered,
        "abstain_zero_accuracy": correct / len(scored),
    }


def _summary(records: list[dict], score_field: str) -> dict:
    return {
        "score_field": score_field,
        "n": len(records),
        "aurc": _aurc(records, score_field),
        "coverage": [
            _at_coverage(records, score_field, coverage)
            for coverage in (50, 70, 90, 100)
        ],
    }


def _group(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    return {
        name: _summary(group, "p_answerable")
        for name, group in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--score-field", default="p_answerable")
    parser.add_argument("--baseline-field", default="max_prob")
    args = parser.parse_args()

    records = _load(args.predictions)
    if not records:
        raise SystemExit("no predictions")

    result = {
        "predictions": str(args.predictions),
        "model": _summary(records, args.score_field),
        "baseline": _summary(records, args.baseline_field),
        "by_type": _group(records, "q_type"),
        "by_reason": _group(records, "reason"),
    }
    payload = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
