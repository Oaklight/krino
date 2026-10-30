#!/usr/bin/env python3
"""Accuracy by state-length quartile on the JevBench hard tier.

Joins a per-item prediction dump to the tier's state lengths and splits the hard
items into quartiles by length, so two scorers can be compared on the same cut.
The quartile boundaries are recomputed from the item set rather than hard-coded,
and printed, because the earlier table in the repository does not say which
checkpoint produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(sys.argv[1])
DUMPS = [Path(arg) for arg in sys.argv[2:]]

items = []
with BENCH.open(encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not str(row.get("id", "")).startswith("hard"):
            continue
        state = row.get("state")
        if not isinstance(state, str):
            state = json.dumps(state)
        items.append({"id": row["id"], "chars": len(state)})

items.sort(key=lambda item: item["chars"])
n = len(items)
bounds = [0, n // 4, n // 2, (3 * n) // 4, n]
groups = [items[bounds[i]:bounds[i + 1]] for i in range(4)]
print(f"hard items: {n}")
print("quartile boundaries (chars):", [f"{g[0]['chars']}-{g[-1]['chars']}" for g in groups])

for dump in DUMPS:
    data = json.loads(dump.read_text(encoding="utf-8"))["predictions"]
    correct = {row["id"]: row["correct"] for row in data}
    accs = []
    for group in groups:
        hits = sum(1 for item in group if correct.get(item["id"]))
        accs.append(hits / len(group) if group else 0.0)
    total = sum(1 for item in items if correct.get(item["id"])) / n
    print(
        f"{dump.name:<40} "
        + "  ".join(f"{value:.3f}" for value in accs)
        + f"   overall {total:.3f}"
    )
