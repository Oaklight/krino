import json
from pathlib import Path
from data.pipeline import load_jsonl

srcs = [Path("data/benchmarks/base_probs_5000.jsonl")] + [
    Path(f"data/benchmarks/base_probs_extra_shard{i}.jsonl") for i in range(4)
]
seen = {}
for path in srcs:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen[rec["id"]] = rec
out = Path("data/benchmarks/base_probs_20000.jsonl")
with out.open("w", encoding="utf-8") as fh:
    for rec in seen.values():
        fh.write(json.dumps(rec) + "\n")
items = load_jsonl(Path("data/benchmarks/residual_train_20000.jsonl"))
missing = [i.id for i in items if i.id not in seen]
print("records", len(seen), "items", len(items), "missing", len(missing))
if missing:
    print("first missing", missing[:5])
