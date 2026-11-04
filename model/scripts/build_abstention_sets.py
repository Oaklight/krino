"""Build constructed abstention sets from an existing TypedQuestion corpus.

The abstention head needs negative examples.  This script derives them from
items that already have a gold label, so no new annotation is required:

- ``none_of_these``: a choice item with the gold option removed.  The gold key
  stays in ``label`` for provenance; the surviving criteria are the choices.
- ``out_of_scope``: a noul question paired with a state from a different
  source.  The answer is False.
- ``insufficient``: a supported noul item with the evidence cut down.  This is
  heuristic by design; the eval stratifies by reason so the score is readable.
- ``no_applicable_tool`` is produced by the tool-call builder, not here.

Every constructed item sets ``answerable=False`` and a closed-set ``reason``,
and strips ``teacher_probs`` so a soft target cannot leak the old label.

Usage:
    python model/scripts/build_abstention_sets.py \
        --items data/benchmarks/residual_train_20000_jev.jsonl \
        --out data/benchmarks/abstention_20k.jsonl \
        --limit-per-reason 1000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import ABSTENTION_REASONS, TypedQuestion  # noqa: E402
from data.pipeline import load_jsonl, save_jsonl  # noqa: E402

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _clone(item: TypedQuestion, **changes) -> TypedQuestion:
    return replace(
        item,
        teacher_probs=None,
        teacher_name=None,
        answerable=False,
        **changes,
    )


def _choice_without_gold(item: TypedQuestion) -> TypedQuestion | None:
    q = item.question
    if q.get("type") != "choice":
        return None
    criteria = q.get("criteria") or {}
    if not isinstance(criteria, dict) or item.label not in criteria:
        return None
    if len(criteria) < 3:
        return None
    surviving = {key: value for key, value in criteria.items() if key != item.label}
    question = dict(q)
    question["criteria"] = surviving
    return _clone(
        item,
        id=f"{item.id}-abstain-none_of_these",
        question=question,
        reason="none_of_these",
    )


def _out_of_scope(item: TypedQuestion, donor_state: str) -> TypedQuestion | None:
    if item.question.get("type") != "noul":
        return None
    if not donor_state or donor_state == item.state:
        return None
    return _clone(
        item,
        id=f"{item.id}-abstain-out_of_scope",
        state=donor_state,
        label=False,
        reason="out_of_scope",
    )


def _insufficient(item: TypedQuestion, fraction: float) -> TypedQuestion | None:
    if item.question.get("type") != "noul" or item.label is not True:
        return None
    sentences = [part for part in SENTENCE.split(item.state) if part.strip()]
    if len(sentences) >= 2:
        kept = sentences[: max(1, int(round(len(sentences) * fraction)))]
        state = " ".join(kept)
    else:
        state = item.state[: max(1, int(len(item.state) * fraction))]
    if not state.strip() or state == item.state:
        return None
    return _clone(
        item,
        id=f"{item.id}-abstain-insufficient",
        state=state,
        label=False,
        reason="insufficient",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument(
        "--reasons",
        default="none_of_these,out_of_scope,insufficient",
        help=f"Comma-separated subset of {ABSTENTION_REASONS}",
    )
    parser.add_argument("--limit-per-reason", type=int, default=1000)
    parser.add_argument("--insufficient-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reasons = [part.strip() for part in args.reasons.split(",") if part.strip()]
    unknown = [reason for reason in reasons if reason not in ABSTENTION_REASONS]
    if unknown:
        raise SystemExit(f"unknown reason(s): {unknown}")
    if not 0.0 < args.insufficient_fraction < 1.0:
        raise SystemExit("--insufficient-fraction must be in (0, 1)")

    items = load_jsonl(args.items)
    rng = random.Random(args.seed)
    by_source: dict[str, list[TypedQuestion]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    selected: list[TypedQuestion] = []
    counts: Counter = Counter()
    seen_ids: set[str] = set()

    def add(candidate: TypedQuestion | None) -> bool:
        if candidate is None or candidate.id in seen_ids:
            return False
        selected.append(candidate)
        seen_ids.add(candidate.id)
        counts[candidate.reason] += 1
        return True

    if "none_of_these" in reasons:
        candidates = [item for item in items if item.question.get("type") == "choice"]
        rng.shuffle(candidates)
        for item in candidates:
            if counts["none_of_these"] >= args.limit_per_reason:
                break
            add(_choice_without_gold(item))

    if "out_of_scope" in reasons:
        candidates = [item for item in items if item.question.get("type") == "noul"]
        rng.shuffle(candidates)
        for item in candidates:
            if counts["out_of_scope"] >= args.limit_per_reason:
                break
            donors = [
                donor
                for source, group in by_source.items()
                if source != item.source
                for donor in group
                if donor.state and donor.state != item.state
            ]
            if donors:
                add(_out_of_scope(item, rng.choice(donors).state))

    if "insufficient" in reasons:
        candidates = [
            item
            for item in items
            if item.question.get("type") == "noul" and item.label is True
        ]
        rng.shuffle(candidates)
        for item in candidates:
            if counts["insufficient"] >= args.limit_per_reason:
                break
            add(_insufficient(item, args.insufficient_fraction))

    save_jsonl(selected, args.out)
    stats = {
        "items": str(args.items),
        "out": str(args.out),
        "reasons": reasons,
        "limit_per_reason": args.limit_per_reason,
        "input": len(items),
        "selected": len(selected),
        "by_reason": dict(counts),
        "by_type": dict(Counter(item.question["type"] for item in selected)),
        "by_source": dict(Counter(item.source for item in selected).most_common(20)),
        "answerable_values": dict(
            Counter(str(item.answerable) for item in selected)
        ),
        "teacher_probs_present": sum(1 for item in selected if item.teacher_probs),
    }
    payload = json.dumps(stats, indent=1, sort_keys=True)
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(payload + "\n")
    print(payload, flush=True)
    print(f"wrote {len(selected)} abstention items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
