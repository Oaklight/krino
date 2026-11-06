#!/usr/bin/env python3
"""Compute the one-page data card for the canonical corpus (#229 Workstream E).

Assembles the primary corpora in the precedence order recorded in
``research/25-corpus-inventory.md``, deduplicates by id, and reports the tier
mix, type mix, choice-K distribution, state-length distribution, and the count
of records eligible for a hard-label loss.

Usage:
    python scripts/corpus_datacard.py \
        --root ~/krino-cpu/data/benchmarks \
        --typed-decisions /tmp/typed_decisions.jsonl \
        --json /tmp/datacard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# Corpus label, filename, precedence.  Synthetic is handled separately.
CANONICAL_FILES = (
    ("target_30k", "label_target_30k_jev.jsonl"),
    ("residual_20k", "residual_train_20000_jev.jsonl"),
    ("typed_decisions", None),  # --typed-decisions path
    ("apigen", "apigen_policy_compliance.jsonl"),
    ("apigen", "apigen_tool_action.jsonl"),
    ("apigen", "apigen_tool_argument.jsonl"),
    ("apigen", "apigen_tool_choice.jsonl"),
    ("apigen", "apigen_tool_next_action.jsonl"),
    ("abstention", "abstention_train_20k.jsonl"),
    ("abstention", "abstention_train_30k.jsonl"),
    ("mmlu_pro", "mmlu_pro_short_reasoning_choice.jsonl"),
    ("long_context", "hotpotqa_long_choice.jsonl"),
    ("long_context", "musique_long_choice.jsonl"),
    ("long_context", "longbench_long_choice.jsonl"),
    ("long_context", "twowiki_long_choice.jsonl"),
    ("race_short", "race_short_passage_choice.jsonl"),
)

CHAR_BUCKETS = (
    (0, 512, "<512"),
    (512, 2_048, "512-2K"),
    (2_048, 8_192, "2K-8K"),
    (8_192, 32_768, "8K-32K"),
    (32_768, None, ">=32K"),
)

K_BUCKETS = (
    (2, 3, "2"),
    (3, 5, "3-4"),
    (5, 9, "5-8"),
    (9, 17, "9-16"),
    (17, 65, "17-64"),
    (65, 256, "65-255"),
)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def bucket(value: int, buckets: tuple) -> str:
    for low, high, label in buckets:
        if high is None and value >= low:
            return label
        if high is not None and low <= value < high:
            return label
    return "other"


def question_k(question: dict) -> int | None:
    q_type = question.get("type")
    if q_type == "choice":
        criteria = question.get("criteria")
        return len(criteria) if isinstance(criteria, dict) else None
    if q_type == "score":
        criteria = question.get("criteria")
        return len(criteria) if isinstance(criteria, list) else None
    return None


def synthetic_items(synth_dir: Path, expand: Any) -> Iterable[dict]:
    for families_path in sorted(synth_dir.glob("*_families.jsonl")):
        domain = families_path.name[: -len("_families.jsonl")]
        variants_path = synth_dir / f"{domain}_variants.jsonl"
        families = list(read_jsonl(families_path))
        variants = list(read_jsonl(variants_path)) if variants_path.exists() else []
        for index, family in enumerate(families):
            variant_doc = variants[index] if index < len(variants) else {}
            yield from expand(family, variant_doc, domain, family.get("_family_idx", index))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--typed-decisions", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    root = args.root.resolve()
    synth_dir = args.synthetic_dir.resolve() if args.synthetic_dir else root / "synthetic"
    from data.provenance import may_drive_hard_label, classify
    from data.synthetic import family_to_typed_questions

    typed_path = args.typed_decisions or (root / "typed_decisions.jsonl")

    seen: set[str] = set()
    by_corpus: Counter = Counter()
    tier: Counter = Counter()
    origin: Counter = Counter()
    types: Counter = Counter()
    choice_k: Counter = Counter()
    score_levels: Counter = Counter()
    lengths: list[int] = []
    lengths_by_type: dict[str, list[int]] = {}
    eligible_ce = 0
    abstain_target = 0
    ineligible_other = 0
    missing_files: list[str] = []

    def add(item: dict, corpus: str) -> None:
        nonlocal eligible_ce, abstain_target, ineligible_other
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return
        # RACE reuses a passage id for several distinct questions.  The id is
        # the split group; (id, question) is the item key.
        if corpus == "race_short":
            key: Any = (item_id, json.dumps(item.get("question") or {}, sort_keys=True))
        else:
            key = item_id
        if key in seen:
            return
        seen.add(key)
        by_corpus[corpus] += 1
        provenance = classify(item, corpus)
        tier[provenance.tier] += 1
        origin[provenance.origin] += 1
        question = item.get("question") or {}
        q_type = str(question.get("type") or "missing")
        types[q_type] += 1
        k = question_k(question)
        if q_type == "choice" and k is not None:
            choice_k[k] += 1
        if q_type == "score" and k is not None:
            score_levels[k] += 1
        state = item.get("state")
        if isinstance(state, str):
            lengths.append(len(state))
            lengths_by_type.setdefault(q_type, []).append(len(state))
        if item.get("answerable") is False:
            abstain_target += 1
        elif may_drive_hard_label(item, corpus):
            eligible_ce += 1
        else:
            ineligible_other += 1

    for corpus, name in CANONICAL_FILES:
        if name is None:
            path = typed_path
        else:
            path = root / name
        if not path.exists():
            missing_files.append(str(path))
            continue
        for item in read_jsonl(path):
            add(item, corpus)
        # Synthetic is last among the corpora that can share ids, matching the
        # precedence recorded in research/25-corpus-inventory.md.
        if corpus == "typed_decisions":
            for item in synthetic_items(synth_dir, family_to_typed_questions):
                add(item, "synthetic")

    if missing_files:
        raise SystemExit("canonical files missing:\n  " + "\n  ".join(missing_files))

    lengths.sort()
    k_hist = {str(k): v for k, v in sorted(choice_k.items())}
    k_buckets: Counter = Counter({label: 0 for _, _, label in K_BUCKETS})
    for k, count in choice_k.items():
        k_buckets[bucket(k, K_BUCKETS)] += count
    k_sorted = sorted(choice_k.elements())
    char_buckets: Counter = Counter()
    for value in lengths:
        char_buckets[bucket(value, CHAR_BUCKETS)] += 1
    by_type_lengths = {
        q_type: {
            "n": len(values),
            "p50": percentile(sorted(values), 0.5),
            "p90": percentile(sorted(values), 0.9),
            "max": max(values) if values else 0,
        }
        for q_type, values in sorted(lengths_by_type.items())
    }

    report = {
        "root": str(root),
        "missing_files": missing_files,
        "total_ids": len(seen),
        "by_corpus": {k: v for k, v in by_corpus.most_common()},
        "tier": dict(tier.most_common()),
        "origin": dict(origin.most_common()),
        "types": dict(types.most_common()),
        "choice_k": {
            "count": sum(choice_k.values()),
            "histogram": k_hist,
            "buckets": dict(k_buckets.most_common()),
            "p50": percentile(k_sorted, 0.5),
            "p90": percentile(k_sorted, 0.9),
            "max": k_sorted[-1] if k_sorted else 0,
        },
        "score_levels": {
            "count": sum(score_levels.values()),
            "histogram": {str(k): v for k, v in sorted(score_levels.items())},
        },
        "state_chars": {
            "n": len(lengths),
            "p10": percentile(lengths, 0.1),
            "p50": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1] if lengths else 0,
            "buckets": dict(char_buckets.most_common()),
            "by_type": by_type_lengths,
        },
        "hard_label": {
            "eligible_ce": eligible_ce,
            "abstain_target": abstain_target,
            "ineligible_other": ineligible_other,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
