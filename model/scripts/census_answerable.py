"""Census the ``answerable=false`` rows of a canonical corpus.

The trainer must not drive a hard-label loss from an abstention row.  Before
that mask is trusted, this script reports what those rows actually hold: how
many there are, by source, and whether an unanswerable row also carries a
well-formed option set with a valid label.  That last shape is the dangerous
one: the mask hides it from the loss, but a later split builder will still see
it as ordinary training data.

Output is a JSON summary.  The corpus path is reduced to its basename in the
output so the committed census carries no machine-specific path.

Usage:
    python model/scripts/census_answerable.py \
        --items /path/to/krino-canonical-243366.jsonl \
        --sha256 07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53 \
        --out results/abstention/canonical_answerable_census.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _options(question: dict) -> list[str]:
    q_type = question.get("type")
    if q_type == "noul":
        return ["false", "true"]
    criteria = question.get("criteria") or {}
    return [str(key) for key in criteria]


def _valid_label(question: dict, label: Any) -> bool:
    q_type = question.get("type")
    if q_type == "noul":
        return isinstance(label, bool) or (
            isinstance(label, str) and label.lower() in {"true", "false"}
        )
    if q_type == "choice":
        return label in (question.get("criteria") or {})
    if q_type == "score":
        try:
            value = float(label)
        except (TypeError, ValueError):
            return False
        levels = len(question.get("criteria") or [])
        return 0 <= value <= levels - 1
    return False


def _well_formed_options(question: dict) -> bool:
    q_type = question.get("type")
    if q_type == "noul":
        return True
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return len(criteria) >= 2 and all(
            isinstance(value, str) and value.strip() for value in criteria.values()
        )
    if isinstance(criteria, list):
        return len(criteria) >= 2 and all(
            isinstance(value, str) and value.strip() for value in criteria
        )
    return False


def _label_shape(label: Any) -> str:
    if isinstance(label, bool):
        return "bool"
    if isinstance(label, (int, float)):
        return "number"
    if isinstance(label, str):
        return "string"
    return type(label).__name__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--max-examples", type=int, default=24)
    args = parser.parse_args(argv)

    answerable_values: Counter = Counter()
    by_source: Counter = Counter()
    by_type: Counter = Counter()
    by_provenance: Counter = Counter()
    by_reason: Counter = Counter()
    label_shapes: Counter = Counter()
    label_values: Counter = Counter()
    option_counts: Counter = Counter()
    option_valid = 0
    label_valid = 0
    label_in_options = 0
    well_formed = 0
    dangerous = 0
    dangerous_by_source: Counter = Counter()
    dangerous_by_reason: Counter = Counter()
    dangerous_by_type: Counter = Counter()
    teacher_probs_present = 0
    reason_without_false = 0
    false_without_reason = 0
    examples: list[dict[str, Any]] = []
    total = 0

    with args.items.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            question = row.get("question") or {}
            label = row.get("label")
            value = row.get("answerable")
            answerable_values["false" if value is False else "true" if value is True else "unset"] += 1
            if value is not False:
                if row.get("reason"):
                    reason_without_false += 1
                continue

            source = str(row.get("source"))
            q_type = str(question.get("type"))
            reason = str(row.get("reason") or "")
            by_source[source] += 1
            by_type[q_type] += 1
            by_provenance[str(row.get("provenance"))] += 1
            by_reason[reason] += 1
            if not reason:
                false_without_reason += 1
            if row.get("teacher_probs"):
                teacher_probs_present += 1

            options = _options(question)
            option_counts[len(options)] += 1
            valid = _valid_label(question, label)
            formed = _well_formed_options(question)
            in_options = valid and (
                q_type in {"noul", "score"} or label in options
            )
            option_valid += int(len(options) >= 2)
            label_valid += int(valid)
            label_in_options += int(in_options)
            well_formed += int(formed)
            label_shapes[_label_shape(label)] += 1
            label_values[repr(label)] += 1

            if formed and valid and in_options:
                dangerous += 1
                dangerous_by_source[source] += 1
                dangerous_by_reason[reason] += 1
                dangerous_by_type[q_type] += 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "source": source,
                            "type": q_type,
                            "reason": reason,
                            "provenance": row.get("provenance"),
                            "label": label if not isinstance(label, str) else label[:40],
                            "label_shape": _label_shape(label),
                            "n_options": len(options),
                            "option_keys": options[:8],
                            "state_prefix": str(row.get("state", ""))[:120],
                        }
                    )

    result = {
        "items": args.items.name,
        "sha256": args.sha256,
        "total": total,
        "answerable_values": dict(answerable_values),
        "answerable_false": {
            "count": answerable_values["false"],
            "by_source": dict(by_source.most_common()),
            "by_type": dict(by_type),
            "by_provenance": dict(by_provenance),
            "by_reason": dict(by_reason.most_common()),
            "teacher_probs_present": teacher_probs_present,
            "option_counts": dict(option_counts.most_common()),
            "rows_with_at_least_two_options": option_valid,
            "label_valid": label_valid,
            "label_in_options": label_in_options,
            "well_formed_options": well_formed,
            "label_shapes": dict(label_shapes),
            "label_values": dict(label_values.most_common(20)),
            "dangerous_well_formed": dangerous,
            "dangerous_by_source": dict(dangerous_by_source.most_common()),
            "dangerous_by_reason": dict(dangerous_by_reason.most_common()),
            "dangerous_by_type": dict(dangerous_by_type),
            "examples": examples,
        },
        "reason_without_answerable_false": reason_without_false,
        "answerable_false_without_reason": false_without_reason,
        "reading": (
            "dangerous_well_formed counts answerable=false rows that also carry "
            "at least two well-formed options and a valid label in those options. "
            "The loss mask hides them; a split builder would not."
        ),
    }
    payload = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
