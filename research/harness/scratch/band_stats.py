import json
from collections import Counter, defaultdict
from pathlib import Path

from data.pipeline import load_jsonl

items = {i.id: i for i in load_jsonl(Path("data/benchmarks/residual_train_20000.jsonl"))}
stats = defaultdict(Counter)
missing = 0
for line in Path("data/benchmarks/base_probs_20000.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    item = items.get(rec["id"])
    if item is None:
        missing += 1
        continue
    kind = rec["q_type"]
    keys, vals = rec["keys"], rec["probs"]
    top = max(range(len(vals)), key=lambda j: vals[j])
    conf = vals[top]
    if kind == "noul":
        correct = (top == 1) == bool(item.label)
    elif kind == "choice":
        correct = str(keys[top]) == str(item.label)
    else:
        correct = int(item.label) == top
    stats[kind]["n"] += 1
    stats[kind]["correct"] += int(correct)
    stats[kind]["conf_correct"] += int(correct and conf >= 0.8)
    stats[kind]["conf_wrong"] += int((not correct) and conf >= 0.8)

total = sum(c["n"] for c in stats.values())
for kind, c in stats.items():
    n = c["n"]
    print(kind, "n", n, "acc %.3f" % (c["correct"] / n),
          "conf_correct %.3f" % (c["conf_correct"] / n),
          "conf_wrong %.3f" % (c["conf_wrong"] / n))
band = sum(c["n"] - c["conf_correct"] - c["conf_wrong"] for c in stats.values())
print("missing", missing, "total", total)
print("band:", band, "/", total, "%.1f%%" % (100 * band / total))
