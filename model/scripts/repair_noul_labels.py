"""Repair noul labels before retraining the unified noul head.

The 30K band-selected noul set has 78.4% teacher-gold agreement, against
89.7% on the 20K set, and the noul arm failed even after the marker readout
unified noul with choice/score.  This script rewrites noul targets without
touching choice or score:

- ``--noul-agreement keep``  drop noul items where the teacher argmax misses
  the gold label (learnability filter);
- ``--noul-agreement exclude``  keep only those misses (a label-noise control);
- ``--noul-gold-mix A``  convex-combine the teacher distribution with the gold
  one-hot: ``(1-A) * teacher + A * onehot(gold)``.  A=1 is hard gold labels;
- ``--noul-band-upper U``  drop noul items whose base p(gold) exceeds U, using
  a ``precompute_base_probs.py`` JSONL.

Augmented noul items are left unchanged: their teacher mass is over the
augmented option set, not ``true``/``false``.  Items with no usable teacher
distribution are kept and counted, so the script reports what it could not
repair instead of silently dropping it.

Usage:
    python model/scripts/repair_noul_labels.py \
        --items data/benchmarks/label_target_30k_jev.jsonl \
        --base-probs data/benchmarks/base_probs_full.jsonl \
        --out data/benchmarks/label_target_30k_jev_noul_repaired.jsonl \
        --noul-agreement keep --noul-gold-mix 0.5 --noul-band-upper 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import TypedQuestion  # noqa: E402
from data.pipeline import load_jsonl, save_jsonl  # noqa: E402

TRUE_KEYS = ("true", "yes", "1")
FALSE_KEYS = ("false", "no", "0")


def teacher_distribution(probs: Any) -> tuple[dict[str, float], str | None] | None:
    """Return ``(distribution, namespace)`` for flat or namespaced teacher probs.

    ``namespace`` is the key that held the distribution (usually ``"jev"``) or
    ``None`` for a flat dict.  The distribution must contain a true/false key.
    """
    if not isinstance(probs, dict) or not probs:
        return None
    # Prefer a named teacher namespace: a few synthetic items carry a flat
    # prior beside the ``jev`` distribution, and the teacher is the nested one.
    for name, value in probs.items():
        if not isinstance(value, dict):
            continue
        nested = {
            str(k): float(v) for k, v in value.items()
            if isinstance(v, (int, float))
        }
        if {key.lower() for key in nested} & set(TRUE_KEYS + FALSE_KEYS):
            return nested, str(name)
    flat = {
        str(k): float(v) for k, v in probs.items()
        if isinstance(v, (int, float))
    }
    if {key.lower() for key in flat} & set(TRUE_KEYS + FALSE_KEYS):
        return flat, None
    return None


def gold_bool(label: Any) -> bool | None:
    if isinstance(label, bool):
        return label
    if isinstance(label, str):
        lowered = label.lower()
        if lowered in TRUE_KEYS:
            return True
        if lowered in FALSE_KEYS:
            return False
    return None


def _key_for(dist: dict[str, float], wanted_true: bool) -> str | None:
    candidates = TRUE_KEYS if wanted_true else FALSE_KEYS
    for candidate in candidates:
        for key in dist:
            if key.lower() == candidate:
                return key
    return None


def teacher_argmax_true(dist: dict[str, float]) -> bool | None:
    key = max(dist, key=dist.get)
    lowered = key.lower()
    if lowered in TRUE_KEYS:
        return True
    if lowered in FALSE_KEYS:
        return False
    return None


def mix_with_gold(
    dist: dict[str, float], gold: bool, weight: float
) -> dict[str, float]:
    """``(1-w) * teacher + w * onehot(gold)`` over the existing keys."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("gold mix weight must be in [0, 1]")
    gold_key = _key_for(dist, gold)
    if gold_key is None:
        raise ValueError("teacher distribution has no true/false key for gold")
    mixed = {key: (1.0 - weight) * value for key, value in dist.items()}
    mixed[gold_key] = mixed[gold_key] + weight
    return mixed


def p_gold_from_base(row: dict, gold: bool) -> float | None:
    keys = row.get("keys") or []
    probs = row.get("probs") or []
    if len(keys) != len(probs):
        return None
    wanted = TRUE_KEYS if gold else FALSE_KEYS
    for wanted_key in wanted:
        for key, prob in zip(keys, probs):
            if str(key).lower() == wanted_key:
                return float(prob)
    return None


def _with_distribution(
    item: TypedQuestion, dist: dict[str, float], namespace: str | None
) -> TypedQuestion:
    if namespace is None:
        probs: dict[str, Any] = dict(dist)
    else:
        probs = dict(item.teacher_probs or {})
        probs[namespace] = dict(dist)
    return replace(item, teacher_probs=probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--base-probs", type=Path, default=None)
    parser.add_argument(
        "--extra-noul",
        type=Path,
        action="append",
        default=None,
        help="Additional labeled corpus whose noul items are appended "
        "(deduplicated by id, augmented noul skipped).  Repeatable.",
    )
    parser.add_argument(
        "--noul-agreement",
        choices=("keep", "exclude", "off"),
        default="off",
        help="keep agreeing noul items, keep only disagreeing ones, or no filter",
    )
    parser.add_argument(
        "--noul-gold-mix",
        type=float,
        default=0.0,
        help="weight on the gold one-hot; 0 leaves teacher targets unchanged",
    )
    parser.add_argument(
        "--noul-band-upper",
        type=float,
        default=None,
        help="drop noul items with base p(gold) above this value",
    )
    args = parser.parse_args()

    if not 0.0 <= args.noul_gold_mix <= 1.0:
        raise SystemExit("--noul-gold-mix must be in [0, 1]")
    if args.noul_band_upper is not None and not 0.0 <= args.noul_band_upper <= 1.0:
        raise SystemExit("--noul-band-upper must be in [0, 1]")
    if args.noul_band_upper is not None and args.base_probs is None:
        raise SystemExit("--noul-band-upper needs --base-probs")

    items = load_jsonl(args.items)
    counts: Counter = Counter()
    if args.extra_noul:
        seen = {item.id for item in items}
        for path in args.extra_noul:
            for item in load_jsonl(path):
                if item.question.get("type") != "noul" or item.question.get(
                    "augmented_options"
                ):
                    counts["extra_noul_skipped"] += 1
                    continue
                if item.id in seen:
                    counts["extra_noul_duplicate"] += 1
                    continue
                seen.add(item.id)
                items.append(item)
                counts["extra_noul_added"] += 1
    base: dict[str, dict] = {}
    if args.base_probs is not None:
        with args.base_probs.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    base[str(row["id"])] = row

    output: list[TypedQuestion] = []
    by_type: Counter = Counter()
    for item in items:
        q = item.question
        q_type = q.get("type")
        by_type[q_type] += 1
        if q_type != "noul" or q.get("augmented_options"):
            output.append(item)
            continue

        gold = gold_bool(item.label)
        parsed = teacher_distribution(item.teacher_probs)
        if gold is None or parsed is None:
            counts["noul_unusable_target"] += 1
            output.append(item)
            continue
        dist, namespace = parsed

        predicted = teacher_argmax_true(dist)
        agree = predicted is not None and predicted == gold
        counts["noul_agree" if agree else "noul_disagree"] += 1

        if args.noul_agreement == "keep" and not agree:
            counts["dropped_agreement"] += 1
            continue
        if args.noul_agreement == "exclude" and agree:
            counts["dropped_agreement"] += 1
            continue

        if args.noul_band_upper is not None:
            row = base.get(item.id)
            p_gold = p_gold_from_base(row, gold) if row else None
            if p_gold is None:
                counts["noul_no_base_prob"] += 1
            elif p_gold > args.noul_band_upper:
                counts["dropped_band"] += 1
                continue

        if args.noul_gold_mix > 0.0:
            dist = mix_with_gold(dist, gold, args.noul_gold_mix)
            counts["noul_gold_mixed"] += 1
        item = _with_distribution(item, dist, namespace)
        output.append(item)

    save_jsonl(output, args.out)
    stats = {
        "items": str(args.items),
        "out": str(args.out),
        "input": len(items),
        "output": len(output),
        "input_by_type": dict(by_type),
        "output_by_type": dict(Counter(item.question["type"] for item in output)),
        "noul_agreement": args.noul_agreement,
        "noul_gold_mix": args.noul_gold_mix,
        "noul_band_upper": args.noul_band_upper,
        "extra_noul": [str(path) for path in (args.extra_noul or [])],
        "counts": dict(counts),
    }
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote {len(output)} items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
