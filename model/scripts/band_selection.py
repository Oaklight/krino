"""Select the learnability band from cached base or current policy probabilities.

Rufus-Air keeps prompt groups whose mean reward lies in ``(0, 0.8]`` and
re-filters online as the policy improves.  Krino's policy is categorical over
at most a few hundred options, so the exact analogue of the group mean is
``p_t(gold)``: the probability the current arm assigns to the gold option.
One forward pass per item gives that value, with no group sampling.

This script applies the rule item by item.  It never loads a model.  Base
probabilities come from ``precompute_base_probs.py``; a mid-training rescore
pass must write the same JSONL format to ``--current-probs``.

Usage:
    python model/scripts/band_selection.py \
        --items data/benchmarks/residual_train_20000.jsonl \
        --base-probs data/benchmarks/base_probs_20000.jsonl \
        --out data/benchmarks/residual_band_20000.jsonl \
        --stats data/benchmarks/residual_band_20000.stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.pipeline import load_jsonl, save_jsonl


def load_probs(path: Path) -> dict[str, dict]:
    """Read ``precompute_base_probs.py`` records keyed by item id."""
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records


def _resolve_index(record: dict, item) -> int | None:
    """Index of the gold option in the probability vector, or ``None``."""
    q_type = item.question["type"]
    if record.get("q_type") not in (None, q_type):
        return None
    probs = record.get("probs") or []
    keys = [str(key) for key in (record.get("keys") or [])]

    if q_type == "noul":
        if len(probs) != 2:
            return None
        index = 1 if bool(item.label) else 0
        want = "yes" if bool(item.label) else "no"
    elif q_type == "choice":
        criteria = list(item.question["criteria"])
        if item.label not in criteria:
            return None
        index = criteria.index(item.label)
        want = str(item.label)
    elif q_type == "score":
        levels = [str(i) for i in range(len(item.question["criteria"]))]
        try:
            index = int(round(float(item.label)))
        except (TypeError, ValueError):
            return None
        if not 0 <= index < len(levels):
            return None
        want = levels[index]
    else:
        return None

    # The cached keys are authoritative; fall back to positional order only
    # when the cache predates the key list.
    if want in keys:
        index = keys.index(want)
    if not 0 <= index < len(probs):
        return None
    return index


def p_gold(record: dict, item) -> float | None:
    index = _resolve_index(record, item)
    if index is None:
        return None
    value = float(record["probs"][index])
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def percentiles(values: list[float], qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for q in qs:
        position = min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))
        out[f"p{q}"] = ordered[position]
    return out


def _bucket(p: float, lower: float, upper: float) -> str:
    if p > upper:
        return "easy"
    if p <= lower:
        return "never"
    return "band"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--base-probs", type=Path, required=True)
    parser.add_argument(
        "--current-probs",
        type=Path,
        default=None,
        help="Mid-training rescore in the same JSONL format; defaults to base",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument(
        "--lower",
        type=float,
        default=0.0,
        help="Drop items with p(gold) <= lower (0.0 drops only exact zeros)",
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=0.8,
        help="Drop items with p(gold) > upper, matching Rufus-Air's 0.8 cut",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.lower < args.upper <= 1.0:
        raise SystemExit("require 0 <= --lower < --upper <= 1")

    items = load_jsonl(args.items)
    if args.limit is not None:
        items = items[: args.limit]
    base = load_probs(args.base_probs)
    current = load_probs(args.current_probs) if args.current_probs else None

    counts = Counter()
    by_type = defaultdict(Counter)
    by_source = defaultdict(Counter)
    p_all = defaultdict(list)
    p_kept = defaultdict(list)
    delta = Counter()

    kept = []
    for index, item in enumerate(items, 1):
        q_type = item.question["type"]
        counts["input"] += 1
        by_type[q_type]["input"] += 1
        by_source[item.source]["input"] += 1

        base_record = base.get(item.id)
        active_record = (current or base).get(item.id)
        if base_record is None or active_record is None:
            counts["missing_probs"] += 1
            by_type[q_type]["missing_probs"] += 1
            by_source[item.source]["missing_probs"] += 1
            continue

        p_base = p_gold(base_record, item)
        p_active = p_gold(active_record, item)
        if p_active is None:
            counts["invalid_label"] += 1
            by_type[q_type]["invalid_label"] += 1
            by_source[item.source]["invalid_label"] += 1
            continue

        p_all[q_type].append(p_active)
        bucket = _bucket(p_active, args.lower, args.upper)
        counts[bucket] += 1
        by_type[q_type][bucket] += 1
        by_source[item.source][bucket] += 1

        if p_base is not None:
            delta["compared"] += 1
            if _bucket(p_base, args.lower, args.upper) != bucket:
                delta["changed"] += 1

        if bucket == "band":
            kept.append(item)
            p_kept[q_type].append(p_active)

        if index % 10000 == 0:
            print(f"  {index}/{len(items)} kept={counts['band']}", flush=True)

    save_jsonl(kept, args.out)

    stats = {
        "items": str(args.items),
        "base_probs": str(args.base_probs),
        "current_probs": str(args.current_probs) if args.current_probs else None,
        "out": str(args.out),
        "rule": (
            "keep if lower < p_t(gold) <= upper; p_t is the current arm's "
            "probability of the gold option, exact for a categorical policy"
        ),
        "thresholds": {"lower": args.lower, "upper": args.upper},
        "counts": {
            "input": counts["input"],
            "kept": counts["band"],
            "easy": counts["easy"],
            "never": counts["never"],
            "missing_probs": counts["missing_probs"],
            "invalid_label": counts["invalid_label"],
        },
        "rates": {
            "kept": counts["band"] / max(1, counts["input"]),
            "easy": counts["easy"] / max(1, counts["input"]),
            "never": counts["never"] / max(1, counts["input"]),
        },
        "by_type": {q_type: dict(counter) for q_type, counter in sorted(by_type.items())},
        "by_source": {
            source: dict(counter) for source, counter in sorted(by_source.items())
        },
        "p_gold": {
            "all": percentiles([p for values in p_all.values() for p in values]),
            "kept": percentiles([p for values in p_kept.values() for p in values]),
            "by_type": {
                q_type: {
                    "all": percentiles(values),
                    "kept": percentiles(p_kept.get(q_type, [])),
                }
                for q_type, values in sorted(p_all.items())
            },
        },
        "band_delta_vs_base": (
            {
                "compared": delta["compared"],
                "changed": delta["changed"],
                "rate": delta["changed"] / max(1, delta["compared"]),
            }
            if current
            else None
        ),
    }
    payload = json.dumps(stats, indent=1, sort_keys=True)
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(payload + "\n")
    print(payload, flush=True)
    print(f"wrote {len(kept)} band items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
