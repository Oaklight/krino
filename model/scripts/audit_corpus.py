"""Audit the corpus a training config actually loads.

Reports, per source: item counts by type, teacher_probs coverage, context
length percentiles, choice K and score level histograms, and the disagreement
between the hard label and the teacher argmax where both exist.

Usage:
    python model/scripts/audit_corpus.py --config model/configs/screen_ettin150m.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from _vendor import yaml  # zerodep YAML; no PyYAML dependency

from data.sources import load_source_items


def _pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.load(args.config.read_text())
    seed = int(config.get("seed", 42))
    rows = []
    totals = collections.Counter()
    type_totals = collections.Counter()
    all_lengths: list[int] = []

    for name, cfg in (config.get("sources") or {}).items():
        if not isinstance(cfg, dict):
            continue
        rng = random.Random(seed)
        train_items, eval_items = load_source_items(name, cfg, rng)
        items = list(train_items) + list(eval_items)
        summary = {
            "source": name,
            "n": len(items),
            "types": dict(collections.Counter(it.question["type"] for it in items)),
            "with_teacher_probs": sum(1 for it in items if getattr(it, "teacher_probs", None)),
            "choice_k": {},
            "score_levels": {},
            "state_chars": {
                "p50": _pct([len(it.state) for it in items], 0.5),
                "p90": _pct([len(it.state) for it in items], 0.9),
                "gt4k": sum(1 for it in items if len(it.state) > 4000),
                "gt8k": sum(1 for it in items if len(it.state) > 8000),
            },
            "label_vs_teacher_disagree": 0,
            "label_vs_teacher_compared": 0,
        }
        k_counter = collections.Counter()
        level_counter = collections.Counter()
        for it in items:
            q = it.question
            if q["type"] == "choice":
                k_counter[len(q["criteria"])] += 1
            elif q["type"] == "score":
                level_counter[len(q["criteria"])] += 1
            tp = getattr(it, "teacher_probs", None)
            if tp and q["type"] in ("choice", "noul"):
                summary["label_vs_teacher_compared"] += 1
                if q["type"] == "choice":
                    keys = list(q["criteria"])
                    teacher = max(keys, key=lambda key: tp.get(key, 0.0))
                    disagree = teacher != it.label
                else:
                    disagree = (tp.get("true", 0.0) >= 0.5) != bool(it.label)
                summary["label_vs_teacher_disagree"] += int(disagree)
        summary["choice_k"] = dict(sorted(k_counter.items()))
        summary["score_levels"] = dict(sorted(level_counter.items()))
        rows.append(summary)
        totals["n"] += len(items)
        totals["with_teacher_probs"] += summary["with_teacher_probs"]
        for qtype, count in summary["types"].items():
            type_totals[qtype] += count
        all_lengths.extend(len(it.state) for it in items)
        print(
            f"{name:16s} n={len(items):6d} teacher={summary['with_teacher_probs']:5d} "
            f"p90chars={summary['state_chars']['p90']:6d} "
            f"K={summary['choice_k']} levels={summary['score_levels']}",
            flush=True,
        )

    payload = {
        "config": str(args.config),
        "totals": dict(totals),
        "types": dict(type_totals),
        "state_chars": {
            "p50": _pct(all_lengths, 0.5),
            "p75": _pct(all_lengths, 0.75),
            "p90": _pct(all_lengths, 0.9),
            "p99": _pct(all_lengths, 0.99),
            "gt4k": sum(1 for v in all_lengths if v > 4000),
            "gt8k": sum(1 for v in all_lengths if v > 8000),
        },
        "sources": rows,
    }
    text = json.dumps(payload, indent=1)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
