"""Re-run the #235 contamination rule and exact-duplicate count on v2."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from data.corpus_audit import (
    ContaminationIndex,
    _record_digest,
    load_jevbench,
    normalize_state,
    word_shingles,
)

CORPUS, JEVBENCH, OUT = sys.argv[1:4]


def main():
    items = load_jevbench(Path(JEVBENCH))
    index = ContaminationIndex(items)
    cache = {}
    hits = []
    by_reason = Counter()
    by_source = Counter()
    seen = set()
    exact_dupes = 0
    rows = 0
    ids = set()
    with open(CORPUS, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows += 1
            rid = str(rec["id"])
            ids.add(rid)
            normalized = normalize_state(rec.get("state", ""))
            digest = _record_digest(
                rid, normalized, rec.get("question") or {}, rec.get("label")
            )
            if digest in seen:
                exact_dupes += 1
            seen.add(digest)
            state_hits = cache.get(normalized)
            if state_hits is None:
                state_hits = [
                    (item.id, item.tier, reason, shared)
                    for item, reason, shared in index.check_state(
                        normalized, word_shingles(normalized)
                    )
                ]
                cache[normalized] = state_hits
            record_hits = []
            exact = index.by_id.get(rid)
            if exact is not None:
                record_hits.append((exact.id, exact.tier, "id", 0))
            record_hits.extend(h for h in state_hits if h[0] != rid)
            if record_hits:
                by_source[rec.get("source") or ""] += 1
                for jev_id, tier, reason, shared in record_hits:
                    by_reason[reason] += 1
                    hits.append(
                        {
                            "record_id": rid,
                            "source": rec.get("source") or "",
                            "jevbench_id": jev_id,
                            "tier": tier,
                            "reason": reason,
                            "shared_shingles": shared,
                        }
                    )
    payload = {
        "corpus": CORPUS,
        "rows": rows,
        "distinct_ids": len(ids),
        "exact_duplicate_rows": exact_dupes,
        "jevbench_items": len(items),
        "hit_rows": sum(by_source.values()),
        "by_source": dict(sorted(by_source.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "hits": hits[:100],
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
