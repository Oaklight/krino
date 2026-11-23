"""Merge the five full-corpus p_base shards into one cache and verify coverage."""
import json
from collections import Counter
from pathlib import Path

from data.pipeline import load_jsonl

root = Path("data/benchmarks")
ids = [item.id for item in load_jsonl(root / "corpus_full.jsonl")]
type_of = {item.id: item.question["type"] for item in load_jsonl(root / "corpus_full.jsonl")}

records: dict[str, dict] = {}
bad = 0
for path in sorted(root.glob("base_probs_full5_shard*.jsonl")):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        records[rec["id"]] = rec

out = root / "base_probs_full.jsonl"
with out.open("w", encoding="utf-8") as handle:
    for item_id in ids:
        rec = records.get(item_id)
        if rec is not None:
            handle.write(json.dumps(rec) + "\n")

missing = [item_id for item_id in ids if item_id not in records]
extra = [item_id for item_id in records if item_id not in type_of]
have = Counter(type_of[i] for i in ids if i in records)
total = Counter(type_of[i] for i in ids)
print("corpus", len(ids), "written", len(ids) - len(missing), "missing", len(missing), "extra", len(extra), "bad lines", bad)
print("by type", {k: f"{have[k]}/{total[k]}" for k in total})
print("out", out)
