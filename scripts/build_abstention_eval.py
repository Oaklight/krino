#!/usr/bin/env python3
"""Build the abstention evaluation instrument from the canonical corpus.

The instrument is a paired set: one answerable donor and one unanswerable
variant per pair.  It is drawn from whole ``(source, group)`` clusters that are
disjoint from the frozen held-out v1 clusters, so no instrument item can share
a group with an acceptance-gate item.

Existing abstention rows in the canonical corpus supply most pairs: each row
whose id is ``<donor>-abstain-<reason>`` is paired with its donor.  When the
disjoint clusters cannot supply the target, the remaining pairs are constructed
with the two evidence edits from ``model/scripts/build_unanswerable_pairs.py``
(``support_removed``, ``entity_corrupted``), which keep the question fixed and
change only the evidence.

The selection artifact is the committed part: ids, ids hash, the clusters that
must be excluded whole from training, and the composition counts.  The paired
rows themselves are written separately with ``--jsonl-out`` and stay beside the
canonical artifact.

Usage:
    python scripts/build_abstention_eval.py \
        --canonical research/krino-canonical-243366.jsonl \
        --heldout model/experiments/canonical/heldout_6400.json \
        --out research/237-abstention-eval-selection.json \
        --jsonl-out research/abstention-eval-11600.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.format import TypedQuestion

ABSTAIN_INFIX = "-abstain-"
DEFAULT_PAIRS = 5800
DEFAULT_ROWS_FLOOR = 8000
CONSTRUCTED_TRANSFORMS = ("support_removed", "entity_corrupted")
TYPED_FIELDS = {field.name for field in fields(TypedQuestion)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _cluster_hash(source: str, group: str | None) -> str:
    return hashlib.sha256(f"{source}\0{group}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_heldout(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("clusters", "ids", "ids_sha256"):
        if key not in payload:
            raise SystemExit(f"held-out artifact {path} has no {key!r}")
    return payload


def donor_of(row: dict[str, Any]) -> str | None:
    record_id = str(row.get("id") or "")
    if ABSTAIN_INFIX not in record_id:
        return None
    return record_id.split(ABSTAIN_INFIX, 1)[0]


def pair_problem(donor: dict[str, Any] | None, variant: dict[str, Any]) -> str | None:
    """Return the first reason a donor/variant pair is not usable."""
    if donor is None:
        return "donor is missing"
    if donor.get("answerable") is False or donor.get("reason"):
        return "donor is not answerable"
    if donor.get("source") != variant.get("source"):
        return "donor and variant differ in source"
    if donor.get("group") != variant.get("group"):
        return "donor and variant differ in group"
    if (donor.get("question") or {}).get("type") != (variant.get("question") or {}).get("type"):
        return "donor and variant differ in question type"
    return None


def existing_pairs(rows: dict[str, dict[str, Any]]) -> tuple[list[dict], collections.Counter]:
    pairs: list[dict] = []
    problems: collections.Counter = collections.Counter()
    for record_id, row in rows.items():
        if row.get("answerable") is not False and not row.get("reason"):
            continue
        donor_id = donor_of(row)
        if donor_id is None:
            problems["variant without an -abstain- id"] += 1
            continue
        donor = rows.get(donor_id)
        problem = pair_problem(donor, row)
        if problem is not None:
            problems[problem] += 1
            continue
        pairs.append(
            {
                "donor": donor,
                "variant": row,
                "reason": row.get("reason"),
                "existing": True,
            }
        )
    return pairs, problems


def cluster_key(row: dict[str, Any]) -> tuple[str, str | None]:
    return (str(row.get("source")), row.get("group"))


def typed(row: dict[str, Any]) -> TypedQuestion:
    return TypedQuestion(**{key: row[key] for key in row if key in TYPED_FIELDS})


def constructible(row: dict[str, Any]) -> bool:
    """Whether the pair builder can derive a variant for this donor row."""
    from model.scripts.build_unanswerable_pairs import (
        NO_SENTENCE_TRANSFORMS,
        SOURCE_FORMAT,
        is_answerable_original,
        split_state,
    )

    if row.get("source") not in SOURCE_FORMAT:
        return False
    if row.get("answerable") is False or row.get("reason"):
        return False
    question = typed(row)
    split = split_state(question)
    return (
        split is not None
        and split.fmt not in NO_SENTENCE_TRANSFORMS
        and is_answerable_original(question)
    )


def construct_pairs(
    candidates: list[dict[str, Any]], need: int, seed: int
) -> tuple[list[dict], collections.Counter]:
    """Build up to ``need`` paired variants from answerable candidate rows."""
    from model.scripts.build_unanswerable_pairs import build_pairs

    usable = [row for row in candidates if constructible(row)]
    if not usable:
        return [], collections.Counter({"no parseable candidate": 1})
    built, _variant_counts, _original_counts, diagnostics = build_pairs(
        [typed(row) for row in usable],
        transforms=CONSTRUCTED_TRANSFORMS,
        per_item_limit=1,
        seed=seed,
    )
    by_id = {row["id"]: row for row in usable}
    pairs: list[dict] = []
    donor: dict[str, Any] | None = None
    for row in built:
        as_dict = row.to_dict()
        if row.answerable is True:
            donor = as_dict
            continue
        original = by_id.get(str(row.id).split("-unans-", 1)[0])
        if original is None or donor is None:
            continue
        pairs.append(
            {
                "donor": original,
                "variant": as_dict,
                "reason": row.reason,
                "existing": False,
            }
        )
        if len(pairs) >= need:
            break
    problems = collections.Counter()
    if diagnostics.get("pair_failures"):
        problems["construction failures"] = len(diagnostics["pair_failures"])
    return pairs, problems


def ordered_pairs(
    pairs: list[dict],
    heldout_clusters: set[tuple[str, str | None]],
    heldout_ids: set[str],
) -> list[dict]:
    eligible = [
        pair
        for pair in pairs
        if cluster_key(pair["variant"]) not in heldout_clusters
        and pair["variant"]["id"] not in heldout_ids
        and pair["donor"]["id"] not in heldout_ids
    ]
    eligible.sort(
        key=lambda pair: (
            _cluster_hash(*cluster_key(pair["variant"])),
            str(pair["variant"]["id"]),
        )
    )
    return eligible


def build(
    canonical: Path,
    heldout_path: Path,
    out: Path,
    jsonl_out: Path | None,
    target_pairs: int,
    rows_floor: int,
    seed: int,
) -> dict[str, Any]:
    heldout = load_heldout(heldout_path)
    heldout_clusters = {
        (str(cluster["source"]), cluster.get("group")) for cluster in heldout["clusters"]
    }
    heldout_ids = {str(record_id) for record_id in heldout["ids"]}

    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(canonical):
        record_id = str(row.get("id") or "")
        if not record_id:
            raise SystemExit("canonical row without an id")
        if record_id in rows:
            raise SystemExit(f"duplicate canonical id {record_id}")
        rows[record_id] = row
    missing_heldout = heldout_ids - rows.keys()
    if missing_heldout:
        raise SystemExit(
            f"{len(missing_heldout)} held-out ids are not in the canonical "
            "artifact; the held-out selection is for a different corpus"
        )

    existing, existing_problems = existing_pairs(rows)
    eligible_existing = ordered_pairs(existing, heldout_clusters, heldout_ids)
    selected = eligible_existing[:target_pairs]

    constructed: list[dict] = []
    construction_problems: collections.Counter = collections.Counter()
    if len(selected) < target_pairs:
        need = target_pairs - len(selected)
        used_clusters = {cluster_key(pair["variant"]) for pair in selected}
        used_donor_ids = {str(pair["donor"]["id"]) for pair in selected}
        candidates = [
            row
            for row in rows.values()
            if row.get("answerable") is not False
            and not row.get("reason")
            and cluster_key(row) not in heldout_clusters
            and row["id"] not in heldout_ids
            and str(row["id"]) not in used_donor_ids
            and constructible(row)
        ]
        # Spread the construction over clusters that no existing pair uses,
        # then allow the rest.
        candidates.sort(
            key=lambda row: (
                cluster_key(row) in used_clusters,
                _cluster_hash(*cluster_key(row)),
                str(row["id"]),
            )
        )
        constructed, construction_problems = construct_pairs(
            candidates[: max(need * 8, 2000)], need, seed
        )
        selected.extend(constructed[:need])

    if len(selected) < target_pairs:
        raise SystemExit(
            f"only {len(selected)} pairs available, target {target_pairs}; "
            f"existing eligible {len(eligible_existing)}, constructed "
            f"{len(constructed)}"
        )

    out_rows: list[dict[str, Any]] = []
    seen_donors: set[str] = set()
    for pair in selected:
        donor = dict(pair["donor"])
        donor_id = str(donor["id"])
        if donor_id not in seen_donors:
            # A donor with two reason variants appears in two pairs but is
            # written once; the analysis clusters the two variants on it.
            donor["answerable"] = True
            out_rows.append(donor)
            seen_donors.add(donor_id)
        out_rows.append(pair["variant"])

    if len(out_rows) < rows_floor:
        raise SystemExit(f"{len(out_rows)} rows, below the {rows_floor}-row floor")

    ids = [str(row["id"]) for row in out_rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted(
            record_id
            for record_id, count in collections.Counter(ids).items()
            if count > 1
        )
        detail = [
            {
                "id": record_id,
                "pairs": [
                    (
                        pair["existing"],
                        pair["variant"]["id"],
                        pair["reason"],
                        pair["donor"].get("group"),
                    )
                    for pair in selected
                    if pair["donor"]["id"] == record_id
                ],
            }
            for record_id in duplicates[:3]
        ]
        raise SystemExit(
            "selected rows contain duplicate ids: "
            + ", ".join(duplicates[:5])
            + f" {detail}"
        )
    leaked_clusters = {
        cluster_key(pair["variant"])
        for pair in selected
        if cluster_key(pair["variant"]) in heldout_clusters
    }
    if leaked_clusters:
        raise SystemExit(
            f"{len(leaked_clusters)} selected clusters are in the held-out set"
        )
    leaked_ids = heldout_ids & set(ids)
    if leaked_ids:
        raise SystemExit(f"{len(leaked_ids)} selected ids are in the held-out set")

    clusters: dict[tuple[str, str | None], dict[str, Any]] = {}
    by_reason: collections.Counter = collections.Counter()
    by_type: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    for pair in selected:
        key = cluster_key(pair["variant"])
        entry = clusters.setdefault(
            key,
            {"source": key[0], "group": key[1], "pairs": 0, "rows": 0},
        )
        entry["pairs"] += 1
        entry["rows"] += 2
        by_reason[str(pair["reason"])] += 1
        by_type[str(pair["variant"].get("question", {}).get("type"))] += 1
        by_source[str(pair["variant"].get("source"))] += 1

    payload = {
        "name": "canonical-abstention-eval",
        "version": 1,
        "canonical": canonical.name,
        "canonical_sha256": _sha256_file(canonical),
        "heldout": {
            "path": (
                heldout_path.as_posix()
                if not heldout_path.is_absolute()
                else heldout_path.name
            ),
            "sha256": _sha256_file(heldout_path),
            "ids_sha256": heldout["ids_sha256"],
            "clusters": len(heldout_clusters),
        },
        "selection": (
            "paired donor/variant rows from whole (source, group) clusters; "
            "clusters are ordered by sha256(source + NUL + group); existing "
            "abstention pairs are taken first, and support_removed / "
            "entity_corrupted pairs are constructed only to reach the target. "
            "Every record in a listed cluster is excluded from training, not "
            "only the emitted rows."
        ),
        "target_pairs": target_pairs,
        "rows_floor": rows_floor,
        "counts": {
            "pairs": len(selected),
            "rows": len(out_rows),
            "distinct_donors": len(seen_donors),
            "existing_pairs": sum(1 for pair in selected if pair["existing"]),
            "constructed_pairs": sum(1 for pair in selected if not pair["existing"]),
            "by_reason": dict(sorted(by_reason.items())),
            "by_type": dict(sorted(by_type.items())),
            "by_source": dict(sorted(by_source.items())),
            "clusters": len(clusters),
        },
        "existing_pair_problems": dict(sorted(existing_problems.items())),
        "construction_problems": dict(sorted(construction_problems.items())),
        "clusters": sorted(
            clusters.values(), key=lambda entry: (entry["source"], str(entry["group"]))
        ),
        "ids_sha256": _ids_sha256(ids),
        "ids": sorted(ids),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w", encoding="utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{len(selected)} pairs, {len(out_rows)} rows, "
        f"{len(seen_donors)} donors; "
        f"{payload['counts']['existing_pairs']} existing, "
        f"{payload['counts']['constructed_pairs']} constructed; "
        f"{len(clusters)} clusters; ids {payload['ids_sha256'][:16]}",
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jsonl-out", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--rows-floor", type=int, default=DEFAULT_ROWS_FLOOR)
    parser.add_argument("--seed", type=int, default=237)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(
        args.canonical,
        args.heldout,
        args.out,
        args.jsonl_out,
        args.pairs,
        args.rows_floor,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
