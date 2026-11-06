#!/usr/bin/env python3
"""Build the canonical corpus under a licence manifest.

The manifest (``data/licence_manifest.json``) lists every source group with a
verdict and the builds that may include it.  This script assembles the canonical
corpus, filters it by the selected build, and writes JSONL.  It never edits the
manifest or the source files.

This script exists for a policy revisit; the current decision keeps every group,
so the `full` build is the training corpus.  See
``research/30-canonical-source-licences.md``.

Builds:
    full                               every group
    exclude_restrictive                permissive + unresolved
    exclude_restrictive_and_unresolved permissive only

The abstention set is constructed from the surviving corpora; the filtered
output here drops abstention records whose own source is excluded, but a full
rebuild of that set is a separate step (its out_of_scope states borrow from
other sources).

Usage:
    python scripts/build_canonical_corpus.py \
        --manifest data/licence_manifest.json --build exclude_restrictive \
        --root data/benchmarks --typed-decisions /path/typed_decisions.jsonl \
        --repo-root . --out /tmp/canonical.jsonl --json /tmp/build.json
    # counts only:
    ... --build full --dry-run --json /tmp/build.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CANONICAL_FILES = (
    ("target_30k", "label_target_30k_jev.jsonl"),
    ("residual_20k", "residual_train_20000_jev.jsonl"),
    ("typed_decisions", None),
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

SEED_SOURCE = {
    "medical_triage": "mednli",
    "legal_judgment": "contractnli",
    "code_review": "codesearchnet",
    "financial_analysis": "tabfact",
    "scientific_reasoning": "arc",
    "content_analysis": "fever",
    "academic_reasoning": "mmlu",
    "commonsense_decision": "commonsenseqa",
    "logical_inference": "logiqa",
    "adversarial_inference": "anli",
    "passage_decision": "boolq",
    "long_document": "quality",
    "multi_hop_reasoning": "hotpotqa",
    "numerical_reasoning": "drop",
}
PURE_GENERATION = {
    "product_categorization",
    "resource_management",
    "sequential_action",
    "safety_moderation",
    "navigation_planning",
    "spatial_reasoning",
    "multi_agent_coordination",
    "education_assessment",
    "game_strategy",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def licence_group(item: dict) -> str:
    source = str(item.get("source") or "missing")
    item_id = str(item.get("id") or "")
    if source == "synthetic":
        parts = item_id.split("-")
        domain = parts[1] if len(parts) > 1 else "?"
        return SEED_SOURCE.get(domain, "project" if domain in PURE_GENERATION else "synthetic_unknown")
    if source.startswith("apigen"):
        return "apigen"
    if source in ("race", "race_short_passage_choice"):
        return "race"
    if source == "hotpotqa_long_choice":
        return "musique" if item_id.startswith("musique-long-") else "hotpotqa"
    if source == "longbench_long_choice":
        return "longbench"
    if source == "2wiki_long_choice":
        return "2wiki"
    if source == "mmlu_pro_short_reasoning_choice":
        return "mmlu_pro"
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/licence_manifest.json"))
    parser.add_argument(
        "--build",
        required=True,
        choices=("full", "exclude_restrictive", "exclude_restrictive_and_unresolved"),
    )
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--typed-decisions", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None, help="output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.dry_run and args.out is None:
        raise SystemExit("--out is required unless --dry-run")

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.synthetic import family_to_typed_questions

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = manifest["groups"]
    included = {
        group
        for group, entry in groups.items()
        if args.build in entry.get("included_in", [])
    }

    root = args.root.resolve()
    synth_dir = args.synthetic_dir.resolve() if args.synthetic_dir else root / "synthetic"
    typed_path = args.typed_decisions or (root / "typed_decisions.jsonl")

    seen: set[Any] = set()
    kept = Counter()
    dropped = Counter()
    unknown_groups: Counter = Counter()
    out_handle = None if args.dry_run else args.out.open("w", encoding="utf-8")

    def emit(item: dict, corpus: str) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return
        if corpus == "race_short":
            key: Any = (item_id, json.dumps(item.get("question") or {}, sort_keys=True))
        else:
            key = item_id
        if key in seen:
            return
        seen.add(key)
        group = licence_group(item)
        q_type = str((item.get("question") or {}).get("type") or "missing")
        if group not in groups:
            unknown_groups[group] += 1
            return
        if group not in included:
            dropped[group] += 1
            return
        kept[q_type] += 1
        kept[f"group:{group}"] += 1
        if out_handle is not None:
            out_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    missing: list[str] = []
    for corpus, name in CANONICAL_FILES:
        path = typed_path if name is None else root / name
        if not path.exists():
            missing.append(str(path))
            continue
        for item in read_jsonl(path):
            emit(item, corpus)
        if corpus == "typed_decisions":
            for families_path in sorted(synth_dir.glob("*_families.jsonl")):
                domain = families_path.name[: -len("_families.jsonl")]
                variants_path = synth_dir / f"{domain}_variants.jsonl"
                families = list(read_jsonl(families_path))
                variants = list(read_jsonl(variants_path)) if variants_path.exists() else []
                for index, family in enumerate(families):
                    variant_doc = variants[index] if index < len(variants) else {}
                    for item in family_to_typed_questions(
                        family, variant_doc, domain, family.get("_family_idx", index)
                    ):
                        emit(item, "synthetic")
    if out_handle is not None:
        out_handle.close()
    if missing:
        raise SystemExit("canonical files missing:\n  " + "\n  ".join(missing))
    if unknown_groups:
        raise SystemExit(
            "groups missing from the manifest (fail closed): "
            + ", ".join(sorted(unknown_groups))
        )

    by_type = {
        kind: kept.get(kind, 0) for kind in ("noul", "choice", "score")
    }
    report = {
        "build": args.build,
        "manifest": str(args.manifest),
        "included_groups": sorted(included),
        "total": sum(by_type.values()),
        "by_type": by_type,
        "by_group": {
            key.split(":", 1)[1]: value
            for key, value in sorted(kept.items())
            if key.startswith("group:")
        },
        "dropped_groups": dict(sorted(dropped.items())),
        "note": "abstention is filtered by its own source; a full rebuild from the "
        "surviving corpora is a separate step because out_of_scope borrows donor states",
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
