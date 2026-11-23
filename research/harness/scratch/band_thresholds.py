import json
from pathlib import Path

from data.pipeline import load_jsonl

items = {i.id: i for i in load_jsonl(Path("data/benchmarks/residual_train_20000.jsonl"))}
rows = []
for line in Path("data/benchmarks/base_probs_20000.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    item = items[rec["id"]]
    keys, vals = rec["keys"], rec["probs"]
    top = max(range(len(vals)), key=lambda j: vals[j])
    conf = vals[top]
    if rec["q_type"] == "noul":
        correct = (top == 1) == bool(item.label)
    elif rec["q_type"] == "choice":
        correct = str(keys[top]) == str(item.label)
    else:
        correct = int(item.label) == top
    rows.append((conf, correct, rec["q_type"]))

for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
    keep = [r for r in rows if not ((r[1] and r[0] >= thr) or ((not r[1]) and r[0] >= thr))]
    print(f"thr {thr}: keep {len(keep)}/{len(rows)} = {100*len(keep)/len(rows):.1f}%")
