"""Carve the frozen 6,400-record held-out from the canonical artifact.

Whole (source, group) clusters. Only single-type clusters with at most 200
records are eligible, so the by-type mix is exact; mixed-type clusters would
carry minority-type records into another type's target. Deterministic order by
SHA-256(source + NUL + group); every record in a selected cluster is held out.

Run against the canonical artifact:

    python model/experiments/canonical/select_heldout.py \
        data/canonical/krino-canonical-243366.jsonl <out.json>

The committed heldout_6400.json was produced by this script on corpus sha256
07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53 and has
ids_sha256 492ee0afd47bfe819dee690127b8cfc2fab8c39fe10c4ebd6293754a53dcf87b.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict

corpus_path, out_path = sys.argv[1], sys.argv[2]
CORPUS_SHA = "07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53"
TARGETS = {"noul": 2987, "choice": 2637, "score": 776}

clusters = defaultdict(lambda: {"ids": [], "types": Counter(), "n": 0})
total = 0
with open(corpus_path) as handle:
    for line in handle:
        record = json.loads(line)
        total += 1
        source = record.get("source") or ""
        group = record.get("group") or ""
        cluster = clusters[(source, group)]
        cluster["ids"].append(record["id"])
        cluster["types"][record["question"]["type"]] += 1
        cluster["n"] += 1

pool = {
    key: value
    for key, value in clusters.items()
    if value["n"] <= 200 and len(value["types"]) == 1
}
print("records", total, "clusters", len(clusters), "single-type pool", len(pool))

selected = []
for q_type, target in TARGETS.items():
    candidates = []
    for key, value in pool.items():
        if next(iter(value["types"])) != q_type:
            continue
        digest = hashlib.sha256((key[0] + "\0" + key[1]).encode()).hexdigest()
        candidates.append((digest, key))
    candidates.sort()
    count = 0
    for _, key in candidates:
        if count >= target:
            break
        selected.append(key)
        count += pool[key]["n"]
    print(q_type, "target", target, "achieved", count)

ids = []
by_type = Counter()
cluster_rows = []
for key in selected:
    value = clusters[key]
    ids.extend(value["ids"])
    by_type.update(value["types"])
    cluster_rows.append({"source": key[0], "group": key[1], "n": value["n"]})

ids = sorted(set(ids))
ids_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
payload = {
    "name": "canonical-heldout-6400",
    "corpus": "data/canonical/krino-canonical-243366.jsonl",
    "corpus_sha256": CORPUS_SHA,
    "selection": "whole (source, group) clusters; only single-type clusters with at most 200 records are eligible, so the by-type mix is exact; per-type targets noul 2987 / choice 2637 / score 776; clusters ordered by sha256(source + NUL + group); all records in a selected cluster held out",
    "targets": TARGETS,
    "counts": {"total": len(ids), "by_type": dict(by_type), "clusters": len(selected)},
    "ids_sha256": ids_sha,
    "clusters": cluster_rows,
    "ids": ids,
}
with open(out_path, "w") as handle:
    json.dump(payload, handle, indent=1)
print("held out", len(ids), "records in", len(selected), "clusters", dict(by_type))
print("ids sha256", ids_sha)
