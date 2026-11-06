#!/usr/bin/env python3
"""Label every canonical record with an ability and a label method (#247).

The input is the JSONL written by ``scripts/build_canonical_corpus.py``.  The
script adds three fields to each record and writes the labelled corpus:

    ability          one of the ten names in ``data/abilities.py``
    label_method     source_rule | heuristic | model
    ability_rule     the rule id that fired, for audit

It also writes a coverage report: the ability x question-form matrix, each
ability's count, share, licence-exclusion counts, label-method split, and the
flags the issue asks for (under 2,000 records, model-only, unsupported).  Two
more flags mark single-source abilities (at least 90% of records from one
source) and licence single points (that source is not permissive), so a
6,000-record ability that is one licence decision from zero is not read as
6,000 records of coverage.  ``--predictions`` joins a per-item prediction file
to the labels and adds accuracy and chance-corrected accuracy by ability.

Gradient eligibility is reported separately because the listwise and pairwise
trainers filter ``TRAINED_TYPES = ("choice", "score")`` and therefore take no
gradient from any noul row.  The per-option residual trainer does train noul
with its fixed two options.  A choice or score item with fewer than two options
has a one-element softmax, whose log is identically zero, and is not eligible
in either trainer.

Usage:
    python scripts/build_canonical_corpus.py --build full \
        --typed-decisions data/benchmarks/typed_decisions.jsonl \
        --out /tmp/canonical.jsonl --json /tmp/canonical.json

    python scripts/label_abilities.py \
        --canonical /tmp/canonical.jsonl \
        --manifest data/licence_manifest.json \
        --out /tmp/canonical_labeled.jsonl \
        --coverage research/247-ability-coverage.json

    python scripts/label_abilities.py --canonical /tmp/canonical.jsonl \
        --sample-out /tmp/handcheck-sample.json --sample-size 10

    python scripts/label_abilities.py --handcheck /tmp/handcheck-filled.json \
        --handcheck-out research/247-handcheck.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.abilities import (
    ABILITIES,
    ABILITY_COOKBOOK,
    ABILITY_DEFINITIONS,
    CLASSIFICATION,
    LABEL_METHODS,
    MODEL,
    classify,
    classify_question,
    is_known_source,
)
from scripts.build_canonical_corpus import licence_group

MIN_RECORDS_FLAG = 2000
SINGLE_SOURCE_SHARE = 0.9
EXCLUSION_BUILDS = ("exclude_restrictive", "exclude_restrictive_and_unresolved")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def option_count(question: dict[str, Any]) -> int:
    q_type = question.get("type")
    if q_type == "noul":
        return 2
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return len(criteria)
    if isinstance(criteria, (list, tuple)):
        return len(criteria)
    return 0


def gradient_eligible(question: dict[str, Any], trainers: str) -> bool:
    """Whether the named trainer can take a gradient from this question.

    ``listwise_pairwise`` mirrors ``TRAINED_TYPES`` in those scripts.  The
    ``residual`` trainer also trains noul, but every question still needs at
    least two options for a non-zero softmax.
    """
    q_type = question.get("type")
    if trainers == "listwise_pairwise" and q_type == "noul":
        return False
    return option_count(question) >= 2


class Coverage:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.groups = manifest["groups"]
        self.included = {
            build: {
                group
                for group, entry in self.groups.items()
                if build in entry.get("included_in", [])
            }
            for build in EXCLUSION_BUILDS
        }
        self.total = 0
        self.matrix: dict[str, collections.Counter[str]] = {
            ability: collections.Counter() for ability in ABILITIES
        }
        self.by_method: dict[str, collections.Counter[str]] = {
            ability: collections.Counter() for ability in ABILITIES
        }
        self.by_verdict: dict[str, collections.Counter[str]] = {
            ability: collections.Counter() for ability in ABILITIES
        }
        self.selective_prediction: collections.Counter[str] = collections.Counter()
        self.excluded: dict[str, collections.Counter[str]] = {
            build: collections.Counter() for build in EXCLUSION_BUILDS
        }
        self.gradient: collections.Counter[str] = collections.Counter()
        self.gradient_by_type: collections.Counter[str] = collections.Counter()
        self.gradient_residual: collections.Counter[str] = collections.Counter()
        self.methods: collections.Counter[str] = collections.Counter()
        self.rules: collections.Counter[str] = collections.Counter()
        self.sources: dict[str, dict[str, Any]] = {}
        self.ability_sources: dict[str, collections.Counter[str]] = {
            ability: collections.Counter() for ability in ABILITIES
        }
        self.ability_groups: dict[str, collections.Counter[str]] = {
            ability: collections.Counter() for ability in ABILITIES
        }
        self.verdicts: collections.Counter[str] = collections.Counter()
        self.types: collections.Counter[str] = collections.Counter()
        self.small_option_items: collections.Counter[str] = collections.Counter()

    def add(self, item: dict[str, Any], ability: str, method: str, rule: str) -> None:
        source = str(item.get("source") or "missing")
        group = licence_group(item)
        if group not in self.groups:
            raise ValueError(f"source {source!r} maps to unknown group {group!r}")
        q_type = str((item.get("question") or {}).get("type") or "missing")
        verdict = str(self.groups[group].get("verdict") or "unknown")
        selective = item.get("answerable") is False or bool(item.get("reason"))

        self.total += 1
        self.types[q_type] += 1
        self.methods[method] += 1
        self.rules[rule] += 1
        self.verdicts[verdict] += 1
        self.matrix[ability][q_type] += 1
        self.matrix[ability]["total"] += 1
        self.by_method[ability][method] += 1
        self.by_verdict[ability][verdict] += 1
        self.ability_sources[ability][source] += 1
        self.ability_groups[ability][group] += 1
        if selective:
            self.selective_prediction[ability] += 1
        for build in EXCLUSION_BUILDS:
            if group not in self.included[build]:
                self.excluded[build][ability] += 1

        question = item.get("question") or {}
        eligible = gradient_eligible(question, "listwise_pairwise")
        if eligible:
            self.gradient[ability] += 1
            self.gradient_by_type[q_type] += 1
        if gradient_eligible(question, "residual"):
            self.gradient_residual[ability] += 1
        if q_type in ("choice", "score") and option_count(question) < 2:
            self.small_option_items[q_type] += 1

        entry = self.sources.setdefault(
            source,
            {
                "records": 0,
                "abilities": collections.Counter(),
                "methods": collections.Counter(),
            },
        )
        entry["records"] += 1
        entry["abilities"][ability] += 1
        entry["methods"][method] += 1

    def report(self, canonical: Path, canonical_sha256: str) -> dict[str, Any]:
        abilities: dict[str, Any] = {}
        for ability in ABILITIES:
            total = self.matrix[ability]["total"]
            flags: list[str] = []
            if total == 0:
                flags.append("unsupported")
            if 0 < total < MIN_RECORDS_FLAG:
                flags.append("under_2000")
            if total > 0 and self.by_method[ability][MODEL] == total:
                flags.append("model_only")
            if total > 0 and self.gradient[ability] == 0:
                flags.append("no_listwise_gradient")
            top_source = None
            single_source = False
            licence_single_point = False
            if total:
                source, source_records = self.ability_sources[ability].most_common(1)[0]
                share = source_records / total
                single_source = share >= SINGLE_SOURCE_SHARE
                group, _ = self.ability_groups[ability].most_common(1)[0]
                verdict = str(self.groups[group].get("verdict") or "unknown")
                licence_single_point = single_source and verdict != "permissive"
                top_source = {
                    "name": source,
                    "records": source_records,
                    "share": round(share, 6),
                    "licence_group": group,
                    "licence_verdict": verdict,
                }
                if single_source:
                    flags.append("single_source")
                if licence_single_point:
                    flags.append("licence_single_point")
            abilities[ability] = {
                "definition": ABILITY_DEFINITIONS[ability],
                "cookbook": ABILITY_COOKBOOK[ability],
                "total": total,
                "share": round(total / self.total, 6) if self.total else 0.0,
                "by_type": {
                    q_type: self.matrix[ability][q_type]
                    for q_type in ("noul", "choice", "score")
                },
                "by_method": {
                    method: self.by_method[ability][method] for method in LABEL_METHODS
                },
                "by_licence_verdict": {
                    verdict: count
                    for verdict, count in sorted(self.by_verdict[ability].items())
                },
                "selective_prediction": self.selective_prediction[ability],
                "after_exclude_restrictive": total
                - self.excluded["exclude_restrictive"][ability],
                "after_exclude_restrictive_and_unresolved": total
                - self.excluded["exclude_restrictive_and_unresolved"][ability],
                "gradient_eligible_listwise": self.gradient[ability],
                "gradient_share": (
                    round(self.gradient[ability] / total, 6) if total else 0.0
                ),
                "top_source": top_source,
                "single_source": single_source,
                "licence_single_point": licence_single_point,
                "flags": flags,
            }

        model_records = self.methods[MODEL]
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "canonical": canonical.name,
            "canonical_sha256": canonical_sha256,
            "canonical_records": self.total,
            "taxonomy": [
                {"name": name, "definition": ABILITY_DEFINITIONS[name]}
                for name in ABILITIES
            ],
            "matrix": {
                ability: {
                    **{
                        q_type: self.matrix[ability][q_type]
                        for q_type in ("noul", "choice", "score")
                    },
                    "total": self.matrix[ability]["total"],
                }
                for ability in ABILITIES
            },
            "abilities": abilities,
            "label_methods": {method: self.methods[method] for method in LABEL_METHODS},
            "model_share": round(model_records / self.total, 6) if self.total else 0.0,
            "types": {
                q_type: {
                    "records": self.types[q_type],
                    "gradient_eligible_listwise": self.gradient_by_type[q_type],
                }
                for q_type in ("noul", "choice", "score")
            },
            "selective_prediction_records": sum(self.selective_prediction.values()),
            "rule_hits": dict(sorted(self.rules.items())),
            "licence_verdicts": dict(sorted(self.verdicts.items())),
            "sources": {
                source: {
                    "records": entry["records"],
                    "abilities": dict(entry["abilities"].most_common()),
                    "methods": dict(entry["methods"]),
                }
                for source, entry in sorted(self.sources.items())
            },
            "small_option_items": dict(self.small_option_items),
            "trainers": {
                "listwise_pairwise": {
                    "trained_types": ["choice", "score"],
                    "note": (
                        "noul rows are filtered out before training; they are "
                        "labelled here but take no gradient"
                    ),
                    "eligible_rows": sum(self.gradient.values()),
                },
                "residual": {
                    "trained_types": ["noul", "choice", "score"],
                    "note": (
                        "noul uses the fixed two-option no/yes set; a "
                        "single-option question would have zero gradient"
                    ),
                    "eligible_rows": sum(self.gradient_residual.values()),
                },
            },
        }


def write_sample(canonical: Path, out: Path, per_ability: int, seed: int) -> None:
    """Reservoir-sample ``per_ability`` rows for each ability."""
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = {ability: [] for ability in ABILITIES}
    seen: collections.Counter[str] = collections.Counter()
    for item in read_jsonl(canonical):
        ability, method, rule = classify(
            str(item.get("source") or ""),
            str(item.get("id") or ""),
            item.get("question") or {},
        )
        seen[ability] += 1
        row = {
            "id": item.get("id"),
            "source": item.get("source"),
            "question_type": (item.get("question") or {}).get("type"),
            "instructions": (item.get("question") or {}).get("instructions"),
            "classifier_ability": ability,
            "label_method": method,
            "rule": rule,
            "human_ability": "",
        }
        bucket = reservoirs[ability]
        if len(bucket) < per_ability:
            bucket.append(row)
        else:
            index = rng.randrange(seen[ability])
            if index < per_ability:
                bucket[index] = row
    sample = [row for ability in ABILITIES for row in reservoirs[ability]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": seed,
                "per_ability": per_ability,
                "abilities": list(ABILITIES),
                "records": sample,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def score_handcheck(path: Path, out: Path | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_ability: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    missing = 0
    for row in payload["records"]:
        human = str(row.get("human_ability") or "").strip()
        if not human:
            missing += 1
            continue
        key = row["classifier_ability"]
        by_ability[key]["total"] += 1
        if human == key:
            by_ability[key]["agree"] += 1
        else:
            by_ability[key][f"human:{human}"] += 1
    total = sum(counter["total"] for counter in by_ability.values())
    agree = sum(counter["agree"] for counter in by_ability.values())
    report = {
        "sample": str(path),
        "per_ability": payload.get("per_ability"),
        "seed": payload.get("seed"),
        "labelled": total,
        "unlabelled": missing,
        "agreement": round(agree / total, 6) if total else 0.0,
        "by_ability": {
            ability: {
                "labelled": counter["total"],
                "agreement": (
                    round(counter["agree"] / counter["total"], 6)
                    if counter["total"]
                    else 0.0
                ),
                "disagreements": {
                    key.split(":", 1)[1]: value
                    for key, value in counter.items()
                    if key.startswith("human:")
                },
            }
            for ability, counter in sorted(by_ability.items())
        },
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def score_predictions(path: Path, id_to_ability: dict[str, str]) -> dict[str, Any]:
    """Join per-item predictions to abilities and report accuracy.

    Each prediction row needs ``id`` and either ``correct`` or a matching
    ``gold``/``pred`` pair.  ``n_options`` is used for the chance correction;
    a noul defaults to two options and a score item with no ``n_options`` is
    counted in accuracy but not in the chance-corrected mean.
    """
    by_ability: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    matched = 0
    unmatched = 0
    for row in read_jsonl(path):
        record_id = str(row.get("id") or "")
        ability = id_to_ability.get(record_id)
        if ability is None:
            unmatched += 1
            continue
        matched += 1
        q_type = str(row.get("q_type") or "unknown")
        correct = row.get("correct")
        if correct is None:
            correct = row.get("gold") == row.get("pred")
        correct = bool(correct)
        n_options = row.get("n_options")
        if n_options is None and q_type == "noul":
            n_options = 2
        chance = 1.0 / n_options if n_options and n_options >= 2 else None
        corrected = (
            ((1.0 if correct else 0.0) - chance) / (1.0 - chance)
            if chance is not None
            else None
        )
        for bucket in (
            by_ability.setdefault(ability, {"n": 0, "correct": 0, "chance": []}),
            by_type.setdefault(q_type, {"n": 0, "correct": 0, "chance": []}),
        ):
            bucket["n"] += 1
            bucket["correct"] += 1 if correct else 0
            if corrected is not None:
                bucket["chance"].append(corrected)

    def summarise(bucket: dict[str, Any]) -> dict[str, Any]:
        n = bucket["n"]
        chance = bucket["chance"]
        return {
            "n": n,
            "accuracy": round(bucket["correct"] / n, 6) if n else 0.0,
            "chance_corrected": (
                round(sum(chance) / len(chance), 6) if chance else None
            ),
            "chance_corrected_n": len(chance),
        }

    return {
        "path": str(path),
        "matched": matched,
        "unmatched": unmatched,
        "metric": (
            "chance_corrected = mean((correct - 1/K) / (1 - 1/K)); "
            "noul K=2 unless n_options is given"
        ),
        "by_ability": {
            ability: summarise(bucket) for ability, bucket in sorted(by_ability.items())
        },
        "by_type": {
            q_type: summarise(bucket) for q_type, bucket in sorted(by_type.items())
        },
    }


def jevbench_ability_map(root: Path) -> dict[str, str]:
    """Map every public JevBench id to an ability from its question form."""
    mapping: dict[str, str] = {}
    for tier in ("easy", "hard", "original"):
        for row in read_jsonl(root / f"{tier}.jsonl"):
            ability, _, _ = classify_question(row.get("question") or {})
            mapping[str(row["id"])] = ability
    return mapping


def demand_report(
    jevbench_map: dict[str, str], corpus_totals: collections.Counter[str]
) -> dict[str, Any]:
    """Describe what the acceptance set can see, ability by ability."""
    counts = collections.Counter(jevbench_map.values())
    total = len(jevbench_map)
    corpus_total = sum(corpus_totals.values())
    accept_classification = counts[CLASSIFICATION] / total if total else 0.0
    corpus_classification = (
        corpus_totals[CLASSIFICATION] / corpus_total if corpus_total else 0.0
    )
    return {
        "population": f"JevBench public, {total} items",
        "source": "classified by question form (classify_question)",
        "by_ability": {ability: counts[ability] for ability in ABILITIES},
        "share": {
            ability: round(counts[ability] / total, 6) if total else 0.0
            for ability in ABILITIES
        },
        "corpus_share": {
            ability: (
                round(corpus_totals[ability] / corpus_total, 6) if corpus_total else 0.0
            )
            for ability in ABILITIES
        },
        "notes": [
            (
                f"The acceptance set is {accept_classification:.1%} Classification "
                f"and the corpus is {corpus_classification:.1%}; the metric "
                "measures the part of the corpus that overlaps with the set, not "
                "the corpus composition."
            ),
            (
                "Scoring has 18 items. A binomial 95% interval at p=0.5 is "
                "about +/-23 points on 18 items, so a change confined to "
                "Scoring cannot move the 231-item total beyond noise by itself; "
                "a held-out split is needed to see a Scoring regression."
            ),
            (
                "Ranking and ML Feature Extraction have zero items in the "
                "acceptance set as well as zero corpus records, so the metric "
                "cannot see those gaps by construction."
            ),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/licence_manifest.json")
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--map-out", type=Path, default=None)
    parser.add_argument("--coverage", type=Path, default=None)
    parser.add_argument("--sample-out", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=247)
    parser.add_argument("--handcheck", type=Path, default=None)
    parser.add_argument("--handcheck-out", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--jevbench-predictions", type=Path, default=None)
    parser.add_argument(
        "--jevbench-root", type=Path, default=Path("data/benchmarks/jevbench")
    )
    args = parser.parse_args(argv)
    if args.predictions is not None and args.jevbench_predictions is not None:
        parser.error("pass --predictions or --jevbench-predictions, not both")

    if args.handcheck is not None:
        report = score_handcheck(args.handcheck, args.handcheck_out)
        print(json.dumps(report, indent=1))
        return 0

    if args.canonical is None:
        parser.error("--canonical is required")
    if args.sample_out is not None:
        write_sample(args.canonical, args.sample_out, args.sample_size, args.seed)
        print(f"wrote {args.sample_out}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coverage = Coverage(manifest)
    canonical_sha256 = _file_sha256(args.canonical)
    handle = None if args.out is None else args.out.open("w", encoding="utf-8")
    map_handle = (
        None if args.map_out is None else args.map_out.open("w", encoding="utf-8")
    )
    seen_ids: set[str] = set()
    id_to_ability: dict[str, str] = {}
    duplicate_ids: set[str] = set()
    unknown: collections.Counter[str] = collections.Counter()
    try:
        for item in read_jsonl(args.canonical):
            source = str(item.get("source") or "missing")
            if not is_known_source(source):
                unknown[source] += 1
                continue
            item_id = str(item.get("id") or "")
            if item_id in seen_ids:
                duplicate_ids.add(item_id)
            seen_ids.add(item_id)
            ability, method, rule = classify(
                source, item_id, item.get("question") or {}
            )
            item["ability"] = ability
            item["label_method"] = method
            item["ability_rule"] = rule
            id_to_ability[item_id] = ability
            coverage.add(item, ability, method, rule)
            if handle is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if map_handle is not None:
                map_handle.write(
                    json.dumps(
                        {
                            "id": item_id,
                            "ability": ability,
                            "label_method": method,
                            "ability_rule": rule,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if handle is not None:
            handle.close()
        if map_handle is not None:
            map_handle.close()

    if unknown:
        raise SystemExit(
            "sources with no ability rule (fail closed): " + ", ".join(sorted(unknown))
        )
    if duplicate_ids:
        raise SystemExit(
            f"{len(duplicate_ids)} duplicate record ids in {args.canonical}; "
            "the sidecar needs unique ids"
        )
    if coverage.total == 0:
        raise SystemExit("no records labelled")

    report = coverage.report(args.canonical, canonical_sha256)
    report["unknown_sources"] = []
    totals = collections.Counter(
        {ability: coverage.matrix[ability]["total"] for ability in ABILITIES}
    )
    try:
        report["demand"] = demand_report(
            jevbench_ability_map(args.jevbench_root), totals
        )
    except FileNotFoundError:
        report["demand"] = {"available": False}
    if args.predictions is not None:
        report["eval"] = score_predictions(args.predictions, id_to_ability)
    if args.jevbench_predictions is not None:
        report["eval"] = score_predictions(
            args.jevbench_predictions, jevbench_ability_map(args.jevbench_root)
        )
        report["eval"]["population"] = "jevbench"
        report["eval"]["ability_source"] = (
            "classify_question: question form only, no source rule"
        )
    if args.map_out is not None:
        map_sha256 = _file_sha256(args.map_out)
        args.map_out.with_name(args.map_out.name + ".sha256").write_text(
            f"{map_sha256}  {args.map_out.name}\n", encoding="utf-8"
        )
        report["sidecar"] = {
            "path": args.map_out.name,
            "records": coverage.total,
            "sha256": map_sha256,
        }
    text = json.dumps(report, indent=1)
    if args.coverage is not None:
        args.coverage.parent.mkdir(parents=True, exist_ok=True)
        args.coverage.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
