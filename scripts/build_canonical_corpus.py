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

Every surviving record is stamped with its `provenance` tier
(``data/provenance.py``) before it is written.  Output is deterministic: fixed
input order, no set iteration, and the sha256 is written beside the file as
``<out>.sha256`` and repeated in the JSON report.

An id must identify one record, because base probabilities and labels are keyed
by id.  When a corpus reuses an id, every record in that corpus gets
``<id>-q<N>`` (N is 1-based in file order) and keeps ``group = <id>``, so the
split still groups the records the original id grouped.  The rule is generic;
today only the RACE short-passage corpus has this shape.

Every choice and score row is annotated with ``label_kind``: ``option`` or
``withheld_gold`` for a choice, ``level`` or ``fractional`` for a score.  The
builder fails on a choice label outside ``criteria`` unless ``answerable`` is
False, and on a score label that is non-numeric or outside the level range; the
failure names the row and the criterion.  Noul rows have no label kind.

Usage:
    python scripts/build_canonical_corpus.py \
        --manifest data/licence_manifest.json --build full \
        --root data/benchmarks --typed-decisions /path/typed_decisions.jsonl \
        --repo-root . --out /tmp/canonical.jsonl --json /tmp/build.json
    # counts only:
    ... --build full --dry-run --json /tmp/build.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

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
    from data.labels import LabelError, annotate
    from data.provenance import tag
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

    seen: set[str] = set()
    kept = Counter()
    dropped = Counter()
    unknown_groups: Counter = Counter()
    kept_ids: set[str] = set()
    out_handle = None if args.dry_run else args.out.open("w", encoding="utf-8")

    canonical_paths = [
        (corpus, typed_path if name is None else root / name)
        for corpus, name in CANONICAL_FILES
    ]
    missing = [str(path) for _, path in canonical_paths if not path.exists()]
    families_paths = sorted(synth_dir.glob("*_families.jsonl"))
    if not families_paths:
        missing.append(str(synth_dir / "*_families.jsonl"))
    if missing:
        raise SystemExit("canonical files missing:\n  " + "\n  ".join(missing))

    def iter_records() -> Iterable[tuple[str, dict]]:
        for corpus, path in canonical_paths:
            for item in read_jsonl(path):
                yield corpus, item
            if corpus == "typed_decisions":
                for families_path in families_paths:
                    domain = families_path.name[: -len("_families.jsonl")]
                    variants_path = synth_dir / f"{domain}_variants.jsonl"
                    families = list(read_jsonl(families_path))
                    variants = list(read_jsonl(variants_path)) if variants_path.exists() else []
                    for index, family in enumerate(families):
                        variant_doc = variants[index] if index < len(variants) else {}
                        for item in family_to_typed_questions(
                            family, variant_doc, domain, family.get("_family_idx", index)
                        ):
                            yield "synthetic", item

    # Pass 1: find corpora whose id is not unique within the corpus.
    id_counts: dict[str, Counter] = {}
    for corpus, item in iter_records():
        item_id = item.get("id")
        if isinstance(item_id, str):
            id_counts.setdefault(corpus, Counter())[item_id] += 1
    suffix_corpora = {
        corpus
        for corpus, counter in id_counts.items()
        if any(count > 1 for count in counter.values())
    }
    collisions = {
        corpus: {
            "ids": sum(1 for count in id_counts[corpus].values() if count > 1),
            "records": sum(count for count in id_counts[corpus].values() if count > 1),
        }
        for corpus in sorted(suffix_corpora)
    }
    ordinals: dict[tuple[str, str], int] = {}

    def emit(item: dict, corpus: str) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return
        if corpus in suffix_corpora:
            ordinal = ordinals.get((corpus, item_id), 0)
            ordinals[(corpus, item_id)] = ordinal + 1
            item = dict(item)
            item["id"] = f"{item_id}-q{ordinal + 1}"
            item["group"] = item_id
            item_id = item["id"]
        if item_id in seen:
            return
        seen.add(item_id)
        group = licence_group(item)
        q_type = str((item.get("question") or {}).get("type") or "missing")
        if group not in groups:
            unknown_groups[group] += 1
            return
        if group not in included:
            dropped[group] += 1
            return
        provenance = tag(item, corpus)
        annotate(item)
        kept[q_type] += 1
        kept[f"group:{group}"] += 1
        kept[f"tier:{provenance.tier}"] += 1
        kind = item.get("label_kind")
        if kind is not None:
            kept[f"kind:{kind}"] += 1
        kept_ids.add(item_id)
        if out_handle is not None:
            out_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Pass 2: emit in canonical order.
    try:
        for corpus, item in iter_records():
            emit(item, corpus)
    except LabelError as exc:
        if out_handle is not None:
            out_handle.close()
        if args.out is not None:
            args.out.unlink(missing_ok=True)
        raise SystemExit(f"label contract violation: {exc}") from exc
    if out_handle is not None:
        out_handle.close()
    if unknown_groups:
        raise SystemExit(
            "groups missing from the manifest (fail closed): "
            + ", ".join(sorted(unknown_groups))
        )

    output_sha256 = None
    if not args.dry_run and args.out is not None:
        digest = hashlib.sha256()
        with args.out.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        output_sha256 = digest.hexdigest()
        args.out.with_name(args.out.name + ".sha256").write_text(
            f"{output_sha256}  {args.out.name}\n", encoding="utf-8"
        )

    by_type = {
        kind: kept.get(kind, 0) for kind in ("noul", "choice", "score")
    }
    by_provenance = {
        tier: kept.get(f"tier:{tier}", 0) for tier in ("gold", "execution", "teacher")
    }
    report = {
        "build": args.build,
        "manifest": str(args.manifest),
        "included_groups": sorted(included),
        "total": sum(by_type.values()),
        "distinct_ids": len(kept_ids),
        "by_type": by_type,
        "by_provenance": by_provenance,
        "by_label_kind": {
            key.split(":", 1)[1]: value
            for key, value in sorted(kept.items())
            if key.startswith("kind:")
        },
        "id_collisions_fixed": collisions,
        "by_group": {
            key.split(":", 1)[1]: value
            for key, value in sorted(kept.items())
            if key.startswith("group:")
        },
        "dropped_groups": dict(sorted(dropped.items())),
        "output": None if args.dry_run else str(args.out),
        "sha256": output_sha256,
        "notes": [
            "The per-source near-duplicate cluster view belongs to the source-corpus "
            "audit. This artifact carries dataset-level source, not the corpus file, "
            "so it supports the cluster totals only."
        ],
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
