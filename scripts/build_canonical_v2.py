"""Build canonical v2 with the per-row near-duplicate rule (policy C).

Remove a non-held-out row iff:
  1. its exact normalized state equals a held-out state, or
  2. its MinHash Jaccard (character 5-grams, 128 permutations, seed 42) to at
     least one held-out state is >= 0.7.

Cluster membership is the transitive closure of the >= 0.7 edge, so a row that
is >= 0.7 to a held-out row is in that row's cluster; the comparison can be
restricted to the held-out states in the same cluster without changing the set.
Held-out rows are never removed.

Usage:
    python scripts/build_canonical_v2.py <corpus.jsonl> <clusters.json> \
        <heldout.json> <out.jsonl> <out-manifest.json>
"""
import hashlib
import json
import multiprocessing
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from data.lsh import MinHash, char_ngrams

NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
NUM_PERM = 128
SEED = 42
NGRAM = 5
THRESHOLD = 0.7
WORKERS = max(1, os.cpu_count() or 1)


def normalize(text):
    value = unicodedata.normalize("NFC", str(text)).casefold()
    return NON_WORD.sub(" ", value).strip()


def state_key(text):
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def signature(text):
    mh = MinHash(num_perm=NUM_PERM, seed=SEED)
    mh.update(char_ngrams(normalize(text), NGRAM))
    return mh.signature


def sig_chunk(payload):
    start, texts = payload
    return [(start + i, signature(t)) for i, t in enumerate(texts)]


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def counts(records):
    return {
        "by_type": dict(sorted(Counter(r["type"] for r in records).items())),
        "by_source": dict(sorted(Counter(r["source"] for r in records).items())),
        "by_label_kind": dict(
            sorted(Counter(r["label_kind"] or "none" for r in records).items())
        ),
    }


def compare(sig, heldout_sigs):
    """Return the best Jaccard estimate against a sequence of signatures."""
    best = 0.0
    for other in heldout_sigs:
        shared = sum(a == b for a, b in zip(sig, other))
        estimate = shared / NUM_PERM
        if estimate > best:
            best = estimate
            if best >= THRESHOLD:
                return best
    return best


def main():
    corpus, clusters_path, heldout_path, out_jsonl, out_manifest = sys.argv[1:6]
    clusters_doc = json.load(open(clusters_path))
    state_to_cluster = clusters_doc["state_to_cluster"]
    heldout = json.load(open(heldout_path))
    heldout_ids = set(heldout["ids"])

    state_text = {}
    rows = []
    with open(corpus, encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = state_key(rec.get("state", ""))
            state_text.setdefault(key, rec.get("state", ""))
            rows.append(
                {
                    "record": rec,
                    "id": rec["id"],
                    "source": rec.get("source") or "",
                    "type": (rec.get("question") or {}).get("type"),
                    "label_kind": rec.get("label_kind"),
                    "key": key,
                    "cluster": state_to_cluster.get(key),
                    "heldout": rec["id"] in heldout_ids,
                }
            )

    heldout_state_keys = {r["key"] for r in rows if r["heldout"]}
    heldout_by_cluster = defaultdict(list)
    for r in rows:
        if r["heldout"] and r["cluster"]:
            heldout_by_cluster[r["cluster"]].append(r["key"])
    heldout_clusters = set(heldout_by_cluster)
    candidates = [
        r for r in rows
        if not r["heldout"] and r["cluster"] in heldout_clusters
    ]

    # Exact-state branch (both inside and outside clusters).
    exact = {r["id"] for r in rows if not r["heldout"] and r["key"] in heldout_state_keys}
    exact_in_cluster = {
        r["id"] for r in rows
        if not r["heldout"] and r["key"] in heldout_state_keys and r["cluster"]
    }
    exact_outside = exact - exact_in_cluster

    # Signatures for candidate states and held-out states.
    needed = sorted({r["key"] for r in candidates} | heldout_state_keys)
    texts = [state_text[key] for key in needed]
    chunk = max(1, min(200, len(texts) // (WORKERS * 4) or 1))
    chunks = [(i, texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
    sigs = [None] * len(texts)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=WORKERS) as pool:
        for result in pool.imap_unordered(sig_chunk, chunks):
            for index, sig in result:
                sigs[index] = sig
    sig_by_key = {key: sigs[i] for i, key in enumerate(needed)}
    heldout_sigs_by_cluster = {
        cid: [sig_by_key[key] for key in keys]
        for cid, keys in heldout_by_cluster.items()
    }

    jaccard_only = set()
    for r in candidates:
        if r["key"] in heldout_state_keys:
            continue
        sig = sig_by_key[r["key"]]
        if compare(sig, heldout_sigs_by_cluster[r["cluster"]]) >= THRESHOLD:
            jaccard_only.add(r["id"])

    removed = exact | jaccard_only
    kept = [r for r in rows if r["id"] not in removed]
    removed_rows = [r for r in rows if r["id"] in removed]

    # Integrity checks.
    assert not (removed & heldout_ids), "removal touches a held-out id"
    assert len(kept) + len(removed) == len(rows) == 243366
    assert len(heldout_ids - {r["id"] for r in kept}) == 0, "held-out id dropped"

    # No surviving training row may be >= threshold to a held-out row.
    surviving_by_cluster = defaultdict(list)
    for r in kept:
        if not r["heldout"] and r["cluster"]:
            surviving_by_cluster[r["cluster"]].append(r)
    direct_exposed = 0
    for cid, members in surviving_by_cluster.items():
        hs = heldout_sigs_by_cluster.get(cid)
        if not hs:
            continue
        for r in members:
            if compare(sig_by_key[r["key"]], hs) >= THRESHOLD:
                direct_exposed += 1
                break
    assert direct_exposed == 0, f"{direct_exposed} rows still above threshold"

    cluster_exposed = {
        r["id"] for r in kept
        if r["heldout"] and r["cluster"] and surviving_by_cluster.get(r["cluster"])
    }

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for r in kept:
            out.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
    file_sha = hashlib.sha256()
    with open(out_jsonl, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            file_sha.update(block)

    training = [r for r in kept if not r["heldout"]]
    manifest = {
        "name": "krino-canonical-v2-per-row",
        "policy": (
            "C: remove a non-held-out row iff its exact normalized state equals "
            "a held-out state, or its MinHash Jaccard to at least one held-out "
            "state is >= 0.7. Cluster mates below the threshold are kept. "
            "Held-out rows are never removed or re-split."
        ),
        "base_artifact": "krino-canonical-243366.jsonl",
        "base_sha256": "8eb3b2a490a04c600bd652ebeed738c02976bb77e731b337efa513805c62e4f6",
        "base_records": 243366,
        "near_dup_parameters": clusters_doc["parameters"],
        "near_dup_threshold": THRESHOLD,
        "removed_records": len(removed),
        "removed_exact_state": len(exact),
        "removed_exact_in_cluster": len(exact_in_cluster),
        "removed_exact_outside_cluster": len(exact_outside),
        "removed_jaccard_only": len(jaccard_only),
        "removed_ids_sha256": sha_ids(removed),
        "kept_records": len(kept),
        "kept_ids_sha256": sha_ids(r["id"] for r in kept),
        "training_rows": len(training),
        "listwise_pairwise_eligible": sum(
            1 for r in kept if r["type"] in ("choice", "score")
        ),
        "cluster_touching_train_rows": len(candidates),
        "below_threshold_cluster_mates_kept": len(candidates) - len(
            [r for r in candidates if r["id"] in removed]
        ),
        "heldout_ids": len(heldout_ids),
        "heldout_ids_sha256": heldout["ids_sha256"],
        "heldout_pairs": len(heldout["clusters"]),
        "heldout_rows_in_v2": sum(1 for r in kept if r["heldout"]),
        "heldout_direct_exposed": direct_exposed,
        "heldout_sharing_cluster_with_below_threshold_row": len(cluster_exposed),
        "removed_counts": counts(removed_rows),
        "kept_counts": counts(kept),
        "artifact_sha256": file_sha.hexdigest(),
        "remove_ids": sorted(removed),
    }
    with open(out_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "remove_ids"}, indent=2))


if __name__ == "__main__":
    main()
