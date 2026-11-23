import json
from pathlib import Path
from data.pipeline import load_jsonl, save_jsonl

cached = {json.loads(l)["id"] for l in Path("data/benchmarks/base_probs_5000.jsonl").read_text().splitlines() if l.strip()}
items = load_jsonl(Path("data/benchmarks/residual_train_20000.jsonl"))
extra = [i for i in items if i.id not in cached]
save_jsonl(extra, Path("data/benchmarks/residual_train_15000_extra.jsonl"))
print("cached", len(cached), "total", len(items), "extra", len(extra))
