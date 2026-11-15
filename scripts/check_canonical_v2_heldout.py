"""Verify the frozen held-out set is unchanged and not reintroduced into training."""
import hashlib
import json
import sys
from collections import Counter, defaultdict

V2, HELDOUT, MANIFEST, OUT = sys.argv[1:5]


def main():
    heldout = json.load(open(HELDOUT))
    heldout_ids = set(heldout["ids"])
    ids_sha = hashlib.sha256("\n".join(sorted(heldout_ids)).encode()).hexdigest()
    manifest = json.load(open(MANIFEST))
    removed = set(manifest["remove_ids"])

    v2_ids = set()
    pair_rows = Counter()
    pair_nonheldout = Counter()
    heldout_pairs = {(c["source"], c["group"]) for c in heldout["clusters"]}
    for line in open(V2, encoding="utf-8"):
        rec = json.loads(line)
        rid = rec["id"]
        v2_ids.add(rid)
        key = (rec.get("source") or "", rec.get("group") or "")
        pair_rows[key] += 1
        if rid not in heldout_ids:
            pair_nonheldout[key] += 1

    training_ids = v2_ids - heldout_ids
    payload = {
        "heldout_ids": len(heldout_ids),
        "heldout_ids_sha256_recomputed": ids_sha,
        "heldout_ids_sha256_recorded": heldout["ids_sha256"],
        "sha_matches": ids_sha == heldout["ids_sha256"],
        "heldout_pairs": len(heldout_pairs),
        "heldout_ids_in_v2": len(heldout_ids & v2_ids),
        "heldout_ids_missing_from_v2": len(heldout_ids - v2_ids),
        "heldout_ids_in_training_side": len(heldout_ids & training_ids),
        "removed_ids_that_are_heldout": len(removed & heldout_ids),
        "v2_rows": len(v2_ids),
        "training_rows": len(training_ids),
        "heldout_pairs_with_nonheldout_rows_in_v2": sum(
            1 for pair in heldout_pairs if pair_nonheldout.get(pair, 0) > 0
        ),
    }
    # How many held-out pairs exist in v2 and with how many rows.
    payload["heldout_pairs_present_in_v2"] = sum(
        1 for pair in heldout_pairs if pair_rows.get(pair, 0) > 0
    )
    payload["heldout_pair_rows_in_v2"] = sum(
        pair_rows.get(pair, 0) for pair in heldout_pairs
    )
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
