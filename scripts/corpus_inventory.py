#!/usr/bin/env python3
"""Inventory every candidate training file in a Krino benchmark directory.

The script is read-only.  It counts, per file: rows, parse errors, distinct ids,
question-type mix, teacher-probability coverage, and hard-label validity.  For
the synthetic corpus it assembles the same item set that ``load_synthetic()``
yields, so family and variant files are reported as they are actually trained,
not as raw JSONL rows.

Usage:
    python scripts/corpus_inventory.py --root data/benchmarks
    python scripts/corpus_inventory.py --root /path/to/benchmarks --json out.json

Run it from the repository root (or pass --repo-root) so that
``data.synthetic.family_to_typed_questions`` can be imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# --- file classification ----------------------------------------------------

SYNTHETIC_SUFFIXES = (
    "_families.jsonl",
    "_variants.jsonl",
    "_labels.jsonl",
    "_confounders.jsonl",
)

# Files whose ids are kept for the cross-file overlap section.  Everything else
# is measured but its id set is dropped after counting, to bound memory.
ID_SET_PREFIXES = (
    "residual",
    "label_target",
    "corpus_full",
    "apigen_",
    "abstention_",
    "mmlu_pro",
    "hotpotqa_long",
    "musique_long",
    "longbench_long",
    "twowiki_long",
    "race_short",
    "typed_decisions",
)

TYPE_SEGMENTS = ("noul", "choice", "score")


def classify(path: Path) -> str:
    name = path.name
    if name.startswith("base_probs"):
        return "probs"
    if name.endswith("_families.jsonl"):
        return "synthetic_families"
    if name.endswith("_variants.jsonl"):
        return "synthetic_variants"
    if name.endswith("_llm_confounders.jsonl"):
        return "synthetic_confounders"
    if name.endswith("_labels.jsonl"):
        return "synthetic_labels"
    return "typed"


def keep_ids(path: Path) -> bool:
    return path.name.startswith(ID_SET_PREFIXES)


def teacher_namespaces(probs: Any) -> list[str]:
    """Return the namespaces holding probabilities: dict keys, or ``flat``."""
    if not isinstance(probs, dict) or not probs:
        return []
    namespaces = [str(k) for k, v in probs.items() if isinstance(v, dict)]
    if namespaces:
        return sorted(namespaces)
    if any(isinstance(v, (int, float)) for v in probs.values()):
        return ["flat"]
    return []


def read_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, line


def label_status(item: dict) -> tuple[bool, str | None]:
    """Return ``(usable, reason)`` for the hard label of a typed item.

    ``reason`` prefixed ``coerce:`` means the label is usable after a type
    normalisation (``"true"`` -> ``True``, ``"2"`` -> ``2.0``); any other
    reason means the label cannot drive a hard-label loss as encoded.
    """
    q = item.get("question")
    if not isinstance(q, dict):
        return False, "question_missing"
    q_type = q.get("type")
    if "label" not in item or item["label"] is None:
        return False, "label_missing"
    label = item["label"]
    if q_type == "noul":
        if isinstance(label, bool):
            return True, None
        if isinstance(label, str) and label.strip().lower() in {"true", "false", "yes", "no", "1", "0"}:
            return True, "coerce:noul_string"
        return False, "noul_unrecognised" if isinstance(label, str) else "noul_not_bool"
    if q_type == "choice":
        criteria = q.get("criteria")
        if not isinstance(criteria, dict):
            return False, "choice_criteria_missing"
        return (label in criteria, None if label in criteria else "choice_not_in_criteria")
    if q_type == "score":
        criteria = q.get("criteria")
        try:
            value = float(label)
        except (TypeError, ValueError):
            return False, "score_not_numeric"
        if not isinstance(criteria, list) or not criteria:
            return False, "score_criteria_missing"
        if not 0.0 <= value <= len(criteria) - 1:
            return False, "score_out_of_0base_range"
        reason = "coerce:score_string" if isinstance(label, str) else None
        return True, reason
    return False, "question_type_unknown"


def scan_probs(path: Path, global_ids: set[str] | None = None) -> dict[str, Any]:
    """Scan a base-model probability file: {id, keys, probs} rows, no labels."""
    report: dict[str, Any] = {
        "path": str(path),
        "kind": "probs",
        "bytes": path.stat().st_size,
        "rows": 0,
        "parse_errors": 0,
        "schema_errors": 0,
        "distinct_ids": 0,
        "duplicate_ids": 0,
        "key_count": Counter(),
    }
    ids: set[str] = set()
    for _, line in read_lines(path):
        report["rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            report["parse_errors"] += 1
            continue
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            report["schema_errors"] += 1
            continue
        if row["id"] in ids:
            report["duplicate_ids"] += 1
        ids.add(row["id"])
        if global_ids is not None:
            global_ids.add(row["id"])
        keys = row.get("keys")
        probs = row.get("probs")
        if isinstance(keys, list) and isinstance(probs, list) and len(keys) == len(probs):
            report["key_count"][str(len(keys))] += 1
        else:
            report["schema_errors"] += 1
    report["distinct_ids"] = len(ids)
    return report


def scan_typed(
    path: Path,
    global_ids: set[str] | None = None,
    collect_ids: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "kind": "typed",
        "bytes": path.stat().st_size,
        "rows": 0,
        "parse_errors": 0,
        "schema_errors": 0,
        "distinct_ids": 0,
        "duplicate_ids": 0,
        "types": Counter(),
        "teacher_rows": 0,
        "teacher_namespaces": Counter(),
        "hard_labels": 0,
        "label_coercions": Counter(),
        "hard_label_invalid": Counter(),
        "augmented_rows": 0,
        "answerable_false": 0,
        "splits": Counter(),
        "sources": Counter(),
    }
    ids: set[str] = set()
    id_rows = 0
    for _, line in read_lines(path):
        report["rows"] += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            report["parse_errors"] += 1
            continue
        if not isinstance(item, dict):
            report["parse_errors"] += 1
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            id_rows += 1
            if item_id in ids:
                report["duplicate_ids"] += 1
            ids.add(item_id)
            if global_ids is not None:
                global_ids.add(item_id)
        else:
            report["schema_errors"] += 1
        question = item.get("question")
        q_type = question.get("type") if isinstance(question, dict) else None
        report["types"][str(q_type or "missing")] += 1
        namespaces = teacher_namespaces(item.get("teacher_probs"))
        if namespaces:
            report["teacher_rows"] += 1
            report["teacher_namespaces"].update(namespaces)
        valid, reason = label_status(item)
        if valid:
            report["hard_labels"] += 1
        if reason:
            if reason.startswith("coerce:"):
                report["label_coercions"][reason[len("coerce:"):]] += 1
            else:
                report["hard_label_invalid"][reason] += 1
        if isinstance(question, dict) and question.get("augmented_options"):
            report["augmented_rows"] += 1
        if item.get("answerable") is False:
            report["answerable_false"] += 1
        report["splits"][str(item.get("split") or "missing")] += 1
        report["sources"][str(item.get("source") or "missing")] += 1
    report["distinct_ids"] = len(ids)
    report["id_rows"] = id_rows
    if collect_ids:
        report["_id_set"] = ids
    return report


def load_label_index(synthetic_dir: Path) -> tuple[set[str], dict[str, set[str]], Counter]:
    """Return labelled ids, ids per teacher namespace, and a row counter."""
    labelled: set[str] = set()
    by_namespace: dict[str, set[str]] = {}
    namespaces: Counter = Counter()
    for path in sorted(synthetic_dir.glob("*_labels.jsonl")):
        for _, line in read_lines(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("id"), str):
                labelled.add(row["id"])
            spaces = teacher_namespaces(row.get("teacher_probs"))
            namespaces.update(spaces)
            for space in spaces:
                by_namespace.setdefault(space, set()).add(str(row.get("id")))
    return labelled, by_namespace, namespaces


def synthetic_items(
    families_path: Path,
    variants_path: Path | None,
    domain: str,
    expand: Any,
) -> tuple[list[dict], int]:
    """Expand one family file the way ``load_synthetic`` does."""
    try:
        families = [json.loads(line) for _, line in read_lines(families_path)]
    except json.JSONDecodeError:
        return [], 1
    variants: list[dict] = []
    if variants_path and variants_path.exists():
        for _, line in read_lines(variants_path):
            try:
                variants.append(json.loads(line))
            except json.JSONDecodeError:
                variants.append({})
    items: list[dict] = []
    errors = 0
    for idx, family in enumerate(families):
        variant_doc = variants[idx] if idx < len(variants) else {}
        family_idx = family.get("_family_idx", idx)
        try:
            items.extend(expand(family, variant_doc, domain, family_idx))
        except (KeyError, TypeError, ValueError):
            errors += 1
    return items, errors


class WarningCounter(logging.Handler):
    """Count warnings the family expander emits for dropped sub-questions."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.samples: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        if len(self.samples) < 5:
            self.samples.append(record.getMessage())


def scan_families(path: Path, synthetic_dir: Path, expand: Any, labelled: set[str]) -> dict[str, Any]:
    domain = path.name[: -len("_families.jsonl")]
    variants_path = synthetic_dir / f"{domain}_variants.jsonl"
    counter = WarningCounter()
    logger = logging.getLogger("data.synthetic")
    logger.addHandler(counter)
    try:
        items, errors = synthetic_items(path, variants_path, domain, expand)
    finally:
        logger.removeHandler(counter)
    ids = [item["id"] for item in items if isinstance(item.get("id"), str)]
    types = Counter(item.get("question", {}).get("type") for item in items)
    variant_types = Counter()
    for item in items:
        item_id = item.get("id", "")
        if "-cf-" in item_id:
            variant_types["counterfactual"] += 1
        elif "-neg-" in item_id:
            variant_types["negation"] += 1
        elif "-para-" in item_id:
            variant_types["paraphrase"] += 1
        else:
            variant_types["base"] += 1
    return {
        "path": str(path),
        "kind": "synthetic_families",
        "domain": domain,
        "bytes": path.stat().st_size,
        "rows": sum(1 for _ in read_lines(path)),
        "expansion_errors": errors,
        "expansion_warnings": counter.count,
        "warning_samples": counter.samples,
        "items": len(items),
        "distinct_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "types": types,
        "item_stages": variant_types,
        "teacher_rows": sum(1 for i in ids if i in labelled),
        "hard_labels": len(items),
    }


def scan_variants(path: Path) -> dict[str, Any]:
    domain = path.name[: -len("_variants.jsonl")]
    counts = Counter()
    errors = 0
    rows = 0
    for _, line in read_lines(path):
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue
        if not isinstance(row, dict):
            errors += 1
            continue
        for stage in ("counterfactual", "negation", "paraphrase"):
            if isinstance(row.get(stage), dict):
                counts[stage] += 1
    return {
        "path": str(path),
        "kind": "synthetic_variants",
        "domain": domain,
        "bytes": path.stat().st_size,
        "rows": rows,
        "parse_errors": errors,
        "stages": counts,
    }


def scan_labels(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "kind": "synthetic_labels",
        "bytes": path.stat().st_size,
        "rows": 0,
        "parse_errors": 0,
        "distinct_ids": 0,
        "duplicate_ids": 0,
        "teacher_namespaces": Counter(),
        "id_types": Counter(),
    }
    ids: set[str] = set()
    for _, line in read_lines(path):
        report["rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            report["parse_errors"] += 1
            continue
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            report["parse_errors"] += 1
            continue
        item_id = row["id"]
        if item_id in ids:
            report["duplicate_ids"] += 1
        ids.add(item_id)
        report["teacher_namespaces"].update(teacher_namespaces(row.get("teacher_probs")))
        report["id_types"][infer_type(item_id)] += 1
    report["distinct_ids"] = len(ids)
    return report


def scan_confounders(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "kind": "synthetic_confounders",
        "bytes": path.stat().st_size,
        "rows": 0,
        "parse_errors": 0,
        "distinct_ids": 0,
        "models": Counter(),
    }
    ids: set[str] = set()
    for _, line in read_lines(path):
        report["rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            report["parse_errors"] += 1
            continue
        if not isinstance(row, dict):
            report["parse_errors"] += 1
            continue
        if isinstance(row.get("id"), str):
            ids.add(row["id"])
        report["models"][str(row.get("model") or "missing")] += 1
    report["distinct_ids"] = len(ids)
    return report


def infer_type(item_id: str) -> str:
    for segment in TYPE_SEGMENTS:
        if f"-{segment}-" in item_id:
            return segment
    return "unknown"


def merge_counter(target: Counter, source: Counter) -> None:
    target.update(source)


def summarize(files: list[dict]) -> dict[str, Any]:
    typed = [f for f in files if f["kind"] == "typed"]
    synthetic = [f for f in files if f["kind"] in {"synthetic_families", "synthetic_labels", "synthetic_variants"}]
    totals = {
        "files": len(files),
        "rows": sum(f.get("rows", 0) for f in files),
        "parse_errors": sum(f.get("parse_errors", 0) for f in files),
        "typed_items": sum(f.get("rows", 0) - f.get("parse_errors", 0) for f in typed),
        "synthetic_items": sum(f.get("items", 0) for f in files if f["kind"] == "synthetic_families"),
    }
    return totals


def overlap_section(id_sets: dict[str, set[str]]) -> dict[str, Any]:
    names = sorted(id_sets)
    pairwise = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = len(id_sets[left] & id_sets[right])
            if shared:
                pairwise[f"{left} & {right}"] = shared
    union: set[str] = set()
    for ids in id_sets.values():
        union |= ids
    return {
        "set_sizes": {name: len(ids) for name, ids in sorted(id_sets.items())},
        "pairwise_intersections": dict(sorted(pairwise.items(), key=lambda kv: -kv[1])),
        "union_distinct_ids": len(union),
    }


def serialise(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(k): v for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, dict):
        return {k: serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(v) for v in value]
    return value


LABEL_TARGET_VARIANTS = (
    "label_target_30k_jev.jsonl",
    "label_target_30k_jev_noul_agree.jsonl",
    "label_target_30k_jev_noul_agree_mix50.jsonl",
    "label_target_30k_jev_noul_mix50.jsonl",
    "label_target_30k_jev_noul_union_agree_mix50.jsonl",
    "label_target_30k_jev_noul_union_mix50.jsonl",
)


def label_target_deltas(root: Path, base_name: str = "label_target_30k_jev.jsonl") -> dict[str, Any] | None:
    """Compare every label_target_30k_* variant against the labelled base.

    Records the full dropped id list, because a variant's cost is exactly the
    ids it removes from the selected primary set.
    """
    base_path = root / base_name
    if not base_path.exists():
        return None
    meta: dict[str, tuple[str, str]] = {}
    for _, line in read_lines(base_path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        question = row.get("question") if isinstance(row.get("question"), dict) else {}
        meta[row["id"]] = (str(row.get("source") or "missing"), str(question.get("type") or "missing"))
    base_ids = set(meta)
    report: dict[str, Any] = {"base": base_name, "base_distinct_ids": len(base_ids), "variants": {}}
    for name in LABEL_TARGET_VARIANTS:
        path = root / name
        if not path.exists():
            continue
        ids: set[str] = set()
        rows = 0
        for _, line in read_lines(path):
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                ids.add(row["id"])
        dropped = sorted(base_ids - ids)
        added = ids - base_ids
        report["variants"][name] = {
            "rows": rows,
            "distinct_ids": len(ids),
            "dropped_count": len(dropped),
            "added_count": len(added),
            "dropped_by_source": dict(Counter(meta[i][0] for i in dropped).most_common()),
            "dropped_by_type": dict(Counter(meta[i][1] for i in dropped).most_common()),
            "dropped_ids": dropped,
            "added_by_source": dict(
                Counter(
                    str(json.loads(line).get("source") or "missing")
                    for line in _lines_for_ids(path, added)
                ).most_common()
            )
            if added
            else {},
        }
    return report


def _lines_for_ids(path: Path, wanted: set[str]) -> Iterable[str]:
    for _, line in read_lines(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id") in wanted:
            yield line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--deltas",
        type=Path,
        default=None,
        help="write id deltas among the label_target_30k_* variants",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    root = args.root.resolve()
    synthetic_dir = args.synthetic_dir.resolve() if args.synthetic_dir else root / "synthetic"
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    expand = None
    if any(synthetic_dir.glob("*_families.jsonl")):
        from data.synthetic import family_to_typed_questions  # noqa: E402

        expand = family_to_typed_questions

    labelled, label_by_namespace, synthetic_namespaces = load_label_index(synthetic_dir)
    files: list[dict[str, Any]] = []
    id_sets: dict[str, set[str]] = {}
    all_ids: set[str] = set()

    for path in sorted(root.glob("*.jsonl")):
        kind = classify(path)
        if kind == "typed":
            collect = keep_ids(path)
            entry = scan_typed(path, all_ids, collect_ids=collect)
            if collect:
                id_sets[path.name] = entry.pop("_id_set")
        elif kind == "probs":
            entry = scan_probs(path, all_ids)
        elif kind == "synthetic_families":
            if expand is None:
                raise SystemExit("family files found but data.synthetic is not importable; pass --repo-root")
            entry = scan_families(path, synthetic_dir, expand, labelled)
        elif kind == "synthetic_variants":
            entry = scan_variants(path)
        elif kind == "synthetic_labels":
            entry = scan_labels(path)
        else:
            entry = scan_confounders(path)
        files.append(entry)

    # Synthetic corpus as the loader assembles it.
    synthetic_totals: dict[str, Any] = {
        "domains": 0,
        "families": 0,
        "items": 0,
        "distinct_ids": 0,
        "types": Counter(),
        "item_stages": Counter(),
        "teacher_rows": 0,
        "expansion_errors": 0,
        "expansion_warnings": 0,
    }
    synthetic_ids: set[str] = set()
    for path in sorted(synthetic_dir.glob("*_families.jsonl")):
        entry = scan_families(path, synthetic_dir, family_to_typed_questions, labelled)
        files.append(entry)
        synthetic_totals["domains"] += 1
        synthetic_totals["families"] += entry["rows"]
        synthetic_totals["items"] += entry["items"]
        synthetic_totals["teacher_rows"] += entry["teacher_rows"]
        synthetic_totals["expansion_errors"] += entry["expansion_errors"]
        synthetic_totals["expansion_warnings"] += entry["expansion_warnings"]
        merge_counter(synthetic_totals["types"], Counter(entry["types"]))
        merge_counter(synthetic_totals["item_stages"], Counter(entry["item_stages"]))
    for path in sorted(synthetic_dir.glob("*_variants.jsonl")):
        files.append(scan_variants(path))
    for path in sorted(synthetic_dir.glob("*_labels.jsonl")):
        files.append(scan_labels(path))
    for path in sorted(synthetic_dir.glob("*_llm_confounders.jsonl")):
        files.append(scan_confounders(path))

    for path in sorted(synthetic_dir.glob("*_families.jsonl")):
        domain = path.name[: -len("_families.jsonl")]
        variants_path = synthetic_dir / f"{domain}_variants.jsonl"
        items, _ = synthetic_items(path, variants_path, domain, family_to_typed_questions)
        for item in items:
            if isinstance(item.get("id"), str):
                synthetic_ids.add(item["id"])
    synthetic_totals["distinct_ids"] = len(synthetic_ids)
    synthetic_totals["teacher_by_namespace"] = {
        space: len(synthetic_ids & ids) for space, ids in sorted(label_by_namespace.items())
    }
    id_sets["synthetic (assembled)"] = synthetic_ids
    all_ids |= synthetic_ids

    if args.deltas:
        deltas = label_target_deltas(root)
        if deltas is None:
            raise SystemExit(f"no label_target_30k_jev.jsonl under {root}")
        args.deltas.parent.mkdir(parents=True, exist_ok=True)
        args.deltas.write_text(json.dumps(deltas, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "root": str(root),
        "files": files,
        "totals": summarize(files),
        "synthetic": synthetic_totals,
        "synthetic_label_namespaces": synthetic_namespaces,
    }
    result["totals"]["distinct_ids_all"] = len(all_ids)
    result["overlaps"] = overlap_section(id_sets)

    payload = serialise(result)
    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| file | kind | rows | distinct ids | noul | choice | score | teacher | coerce | hard-label invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in report["files"]:
        types = entry.get("types", {})
        if entry["kind"] == "typed":
            dup = entry.get("duplicate_ids", 0)
            rows = entry["rows"]
            if dup:
                rows = f"{entry['rows']} ({dup} dup)"
            lines.append(
                f"| {Path(entry['path']).name} | typed | {rows} | {entry['distinct_ids']} "
                f"| {types.get('noul', 0)} | {types.get('choice', 0)} | {types.get('score', 0)} "
                f"| {entry['teacher_rows']} | {sum(entry['label_coercions'].values())} "
                f"| {sum(entry['hard_label_invalid'].values())} |"
            )
        elif entry["kind"] == "probs":
            lines.append(
                f"| {Path(entry['path']).name} | base-probs | {entry['rows']} "
                f"| {entry['distinct_ids']} | | | | | | {entry['schema_errors']} |"
            )
        elif entry["kind"] == "synthetic_families":
            lines.append(
                f"| synthetic/{Path(entry['path']).name} | families | {entry['rows']} "
                f"| {entry['distinct_ids']} | {types.get('noul', 0)} | {types.get('choice', 0)} "
                f"| {types.get('score', 0)} | {entry['teacher_rows']} | | {entry['expansion_errors']} |"
            )
        else:
            lines.append(
                f"| synthetic/{Path(entry['path']).name} | {entry['kind']} | {entry['rows']} "
                f"| {entry.get('distinct_ids', '')} | | | | | | |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
