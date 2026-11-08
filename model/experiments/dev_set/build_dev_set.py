"""Carve the 800-item judgement-format development set (#298).

Two sources, both already labelled, nothing new annotated:

* the frozen held-out (:file:`model/experiments/canonical/heldout_6400.json`),
  sampled by **whole (source, group) clusters** so the set is not inflated by
  correlated rows, and
* :file:`data/benchmarks/openjev/authored_holdout.jsonl`, all of it.

Cluster-level sampling matters more than item-level here. The held-out holds
6,424 records in 928 clusters; drawing individual rows would let a 200-row
cluster supply a quarter of the set and make a 545-item sample behave like a
much smaller one. Every selected cluster is recorded in the manifest so the
promotion evaluation can exclude it.

The output is a self-contained Jsonl of ``TypedQuestion`` rows, committed, so a
later reader can reproduce the set without the corpus.

Usage::

    python model/experiments/dev_set/build_dev_set.py \\
        data/canonical/krino-canonical-243366.jsonl \\
        --out data/benchmarks/dev/dev_800.jsonl \\
        --manifest data/benchmarks/dev/dev_800.json

The defaults reproduce the first pass.  An all-held-out set with a different
stratification and a cluster-size cap uses::

    python model/experiments/dev_set/build_dev_set.py \\
        data/canonical/krino-canonical-243366.jsonl \\
        --out data/benchmarks/dev/dev_800.jsonl \\
        --manifest data/benchmarks/dev/dev_800.json \\
        --targets noul=375,choice=330,score=95 --no-authored \\
        --max-cluster-share 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

# Held-out records to draw, by type, proportional to the canonical corpus
# (noul 113,594 / choice 100,257 / score 29,515).
HELDOUT_TARGETS = {"noul": 256, "choice": 224, "score": 65}
# The held-out holds 6,424 ids; 6,254 of them are scored by the promotion
# evaluation under the current label-index rule.  The manifest records what the
# promotion set becomes once the reserved clusters are excluded.
HELDOUT_SCORED_ROWS = 6254
AUTHORED_PATH = Path("data/benchmarks/openjev/authored_holdout.jsonl")
HELDOUT_PATH = Path("model/experiments/canonical/heldout_6400.json")


def cluster_key(source: str, group: str) -> str:
    return f"{source}\x00{group}"


def order_key(source: str, group: str) -> str:
    return hashlib.sha256(cluster_key(source, group).encode()).hexdigest()


def load_heldout_ids(path: Path) -> tuple[set[str], list[dict]]:
    payload = json.loads(path.read_text())
    clusters = payload["clusters"]
    ids = set(payload.get("ids") or [])
    return ids, clusters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authored", type=Path, default=AUTHORED_PATH)
    parser.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    parser.add_argument(
        "--targets",
        default=None,
        help="type counts, e.g. noul=375,choice=330,score=95; defaults to HELDOUT_TARGETS",
    )
    parser.add_argument(
        "--no-authored",
        action="store_true",
        help="omit data/benchmarks/openjev/authored_holdout.jsonl from the set",
    )
    parser.add_argument(
        "--max-cluster-share",
        type=float,
        default=None,
        help="skip clusters larger than this fraction of the target total, so one "
        "cluster cannot set the design effect",
    )
    args = parser.parse_args()

    targets = dict(HELDOUT_TARGETS)
    if args.targets:
        targets = {}
        for part in args.targets.split(","):
            name, _, value = part.partition("=")
            targets[name.strip()] = int(value)
        unknown = sorted(set(targets) - set(HELDOUT_TARGETS))
        if unknown:
            parser.error(f"unknown type in --targets: {', '.join(unknown)}")
    max_cluster_rows = None
    if args.max_cluster_share is not None:
        max_cluster_rows = int(args.max_cluster_share * sum(targets.values()))

    heldout_ids, heldout_clusters = load_heldout_ids(args.heldout)
    wanted = {cluster_key(c["source"], c["group"]) for c in heldout_clusters}

    # One pass: keep only held-out records, grouped by cluster.
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    n_corpus = 0
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            n_corpus += 1
            record = json.loads(line)
            key = cluster_key(record.get("source") or "", record.get("group") or "")
            if key in wanted:
                by_cluster[key].append(record)

    assert sum(len(v) for v in by_cluster.values()) == len(heldout_ids), (
        "held-out ids in the corpus do not match the manifest; wrong corpus or wrong manifest"
    )

    # Stratify by the cluster's type. Held-out clusters are single-type by
    # construction (select_heldout.py requires it), so one type per cluster.
    pool: dict[str, list[str]] = defaultdict(list)
    for key, records in by_cluster.items():
        types = {r["question"]["type"] for r in records}
        if len(types) != 1:
            continue  # defensive: the held-out selection forbids these
        pool[types.pop()].append(key)

    selected: list[str] = []
    skipped_oversized: list[str] = []
    for q_type, target in targets.items():
        keys = sorted(pool[q_type], key=lambda k: hashlib.sha256(k.encode()).hexdigest())
        taken = 0
        for key in keys:
            if taken >= target:
                break
            if max_cluster_rows is not None and len(by_cluster[key]) > max_cluster_rows:
                skipped_oversized.append(key)
                continue
            selected.append(key)
            taken += len(by_cluster[key])

    rows = [r for key in selected for r in by_cluster[key]]

    authored: list[dict] = []
    if not args.no_authored:
        authored = [
            json.loads(line)
            for line in args.authored.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rows.extend(authored)

    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids across sources"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = Counter(r["question"]["type"] for r in rows)
    by_source = Counter((r.get("source") or "?") for r in rows)
    heldout_row_ids = {r["id"] for r in rows if r["id"] not in {a["id"] for a in authored}}
    sources: dict[str, dict] = {
        "frozen_heldout": {
            "items": len(heldout_row_ids),
            "clusters": len(selected),
            "clusters_reserved": sorted(selected),
        },
    }
    if not args.no_authored:
        sources["authored_holdout"] = {
            "items": len(authored),
            "clusters": len({a.get("group") for a in authored}),
        }
    manifest = {
        "name": "krino-dev-800",
        "purpose": "development instrument for the judgement readout; JevBench public 231 remains acceptance",
        "sources": sources,
        "counts": {
            "total": len(rows),
            "by_type": dict(by_type),
            "by_source": dict(by_source),
            "by_source_top": dict(by_source.most_common(10)),
            "distinct_sources": len(by_source),
            "targets": targets,
        },
        "cluster_sizes": {key: len(by_cluster[key]) for key in sorted(selected)},
        "max_cluster_rows": max_cluster_rows,
        "clusters_skipped_oversized": sorted(skipped_oversized),
        "promotion": {
            "heldout_scored_rows": HELDOUT_SCORED_ROWS,
            "reserved_rows": len(heldout_row_ids),
            "scored_rows_after_reservation": HELDOUT_SCORED_ROWS - len(heldout_row_ids),
        },
        "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "corpus_records_scanned": n_corpus,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"corpus {n_corpus} records, held-out clusters {len(by_cluster)}")
    print(f"held-out selected {len(selected)} clusters / {len(heldout_row_ids)} rows")
    print(f"authored {len(authored)} rows")
    if skipped_oversized:
        print(f"oversized clusters skipped {len(skipped_oversized)}")
    print(f"promotion rows after reservation {HELDOUT_SCORED_ROWS - len(heldout_row_ids)}")
    print(f"total {len(rows)} rows, types {dict(by_type)}, sources {len(by_source)}")
    print(f"ids_sha256 {manifest['ids_sha256']}")
    print(f"written {args.out} and {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
