"""Seed new shard files from the records already computed, so resume skips them."""
import json
from pathlib import Path

from data.pipeline import load_jsonl

ROOT = Path("data/benchmarks")
ids = [item.id for item in load_jsonl(ROOT / "corpus_full.jsonl")]
index = {item_id: i for i, item_id in enumerate(ids)}

records = {}
for path in sorted(ROOT.glob("base_probs_full_shard*.jsonl")):
    bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if rec.get("id") in index:
            records[rec["id"]] = rec
    print(path.name, "bad lines", bad)
print("unique records", len(records), "of", len(ids))

shards = 5
handles = [(ROOT / f"base_probs_full5_shard{i}.jsonl").open("w", encoding="utf-8") for i in range(shards)]
written = [0] * shards
for item_id, rec in records.items():
    i = index[item_id] % shards
    handles[i].write(json.dumps(rec) + "\n")
    written[i] += 1
for handle in handles:
    handle.close()
print("seeded per shard", written, "total", sum(written))
