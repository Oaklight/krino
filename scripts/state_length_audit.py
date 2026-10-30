#!/usr/bin/env python3
"""Where the JevBench hard tier sits in the training corpus's length distribution.

`research/13` measures a long-state wall on the hard tier: accuracy falls about
30 points from the shortest to the longest quartile of hard-tier states, for both
the trained head and the untouched backbone's native readout. That rules out the
head and points at how much of a state the backbone uses -- but it does not say
whether the corpus ever put a state that long in front of the model.

This script answers that with the files themselves. It reads the `state` field of
every source the training config names, keeps the rows the training split would
see, splits by question type, and compares against JevBench's hard tier.

The unit matters: a long state asked as a *choice* is the hard tier's shape, and
the corpus's long states are mostly noul (contractnli, multirc, tabfact, boolq).
Reporting only the marginal distribution over all types would therefore describe
a region the model does train on while missing the one it does not.

Usage:
    python scripts/state_length_audit.py --data-root data/benchmarks --config \\
        model/configs/r2_reranker4b.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _length(state) -> int:
    return len(state if isinstance(state, str) else json.dumps(state))


def _iter_rows(path: Path, cap: int | None):
    """Rows of a source file in the train split, at most ``cap`` of them.

    ``cap`` mirrors the config's ``max_train``, because the corpus the model
    actually sees is the capped one -- an uncapped census would describe a
    distribution no run drew from.
    """
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") not in (None, "train"):
                continue
            yield row
            seen += 1
            if cap is not None and seen >= cap:
                return


def _type(row) -> str:
    question = row.get("question") or {}
    return str(question.get("type") or "?")


def _pct(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model/configs/r2_reranker4b.yaml"),
        help="the training config whose sources and max_train caps define the corpus",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="held-out file to compare against (default: <data-root>/jevbench.jsonl)",
    )
    args = parser.parse_args()
    bench = args.benchmark or args.data_root / "jevbench.jsonl"

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    sources = cfg.get("sources") or {}

    print(f"{'source':<20} {'type':<7} {'n':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}")
    print("-" * 66)
    choice: list[int] = []
    every: list[int] = []
    for name, source_cfg in sources.items():
        if not isinstance(source_cfg, dict) or name == "synthetic":
            # Synthetic states are assembled by data/pipeline.py at load time;
            # the per-domain files are not what the model sees, so they are
            # excluded rather than measured wrongly.
            continue
        path = args.data_root / str(source_cfg.get("path") or f"{name}.jsonl").split("benchmarks/")[-1]
        if not path.exists():
            continue
        cap = source_cfg.get("max_train")
        per_type: dict[str, list[int]] = {}
        for row in _iter_rows(path, cap):
            value = _length(row.get("state"))
            per_type.setdefault(_type(row), []).append(value)
            every.append(value)
            if _type(row) == "choice":
                choice.append(value)
        for kind, values in sorted(per_type.items()):
            print(
                f"{name:<20} {kind:<7} {len(values):>7} {_pct(values, .5):>7} "
                f"{_pct(values, .9):>7} {_pct(values, .99):>7} {max(values):>7}"
            )

    print()
    print(f"{'corpus':<20} {'any':<7} {len(every):>7} {_pct(every, .5):>7} "
          f"{_pct(every, .9):>7} {_pct(every, .99):>7} {max(every):>7}")
    print(f"{'corpus':<20} {'choice':<7} {len(choice):>7} {_pct(choice, .5):>7} "
          f"{_pct(choice, .9):>7} {_pct(choice, .99):>7} {max(choice):>7}")

    if not bench.exists():
        print(f"\n{bench} not found; skipping the comparison.")
        return 0

    tiers: dict[str, dict[str, list[int]]] = {}
    with bench.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tier = str(row.get("id", "")).split("-")[0] or "?"
            kind = _type(row)
            value = _length(row.get("state"))
            entry = tiers.setdefault(tier, {"all": [], "choice": []})
            entry["all"].append(value)
            if kind == "choice":
                entry["choice"].append(value)

    print("\nJevBench, by id prefix (the tier):")
    for tier, entry in sorted(tiers.items()):
        values = entry["all"]
        choices = entry["choice"]
        print(
            f"  {tier:<10} all n={len(values):<4} p50={_pct(values, .5):<6} "
            f"max={max(values):<6} | choice n={len(choices):<4} "
            f"p25={_pct(choices, .25):<6} p50={_pct(choices, .5):<6} "
            f"max={max(choices) if choices else 0}"
        )

    hard = tiers.get("hard", {}).get("choice") or []
    if not choice or not hard:
        return 0
    print("\nShare of the corpus's choice items at least as long as a hard-tier one:")
    for label, threshold in (
        ("p25", _pct(hard, 0.25)),
        ("p50", _pct(hard, 0.50)),
        ("p75", _pct(hard, 0.75)),
    ):
        over = sum(1 for value in choice if value >= threshold)
        print(
            f"  >= hard {label} ({threshold} chars): "
            f"{over}/{len(choice)} = {100.0 * over / len(choice):.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
