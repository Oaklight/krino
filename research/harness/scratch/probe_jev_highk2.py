"""Larger Jev high-K probe on banking77 (diagnostic only, not acceptance)."""
import random
import time
from pathlib import Path

from jev_client import JevClient

from data.pipeline import load_jsonl

items = [
    i for i in load_jsonl(Path("data/benchmarks/corpus_full.jsonl"))
    if i.question["type"] == "choice" and len(i.question["criteria"]) == 80
][:30]
print("items", len(items), flush=True)
rng = random.Random(7)
client = JevClient()
stats = {}
try:
    for item in items:
        crit = item.question["criteria"]
        gold = str(item.label)
        if gold not in crit:
            continue
        others = [k for k in crit if k != gold]
        for k in (2, 5, 20, 80):
            keep = [gold] + rng.sample(others, k - 1)
            qid = f"{item.id}-K{k}"
            question = {
                "type": "choice",
                "instructions": item.question["instructions"],
                "criteria": {key: crit[key] for key in keep},
            }
            start = time.monotonic()
            try:
                resp = client.ask(item.state, {qid: question})
                elapsed = time.monotonic() - start
                ans = resp.get("answers", {}).get(qid, {})
                probs = ans.get("probabilities", {})
                argmax = max(probs, key=probs.get) if probs else None
                bucket = stats.setdefault(k, {"n": 0, "hit": 0, "lat": [], "covered": 0})
                bucket["n"] += 1
                bucket["hit"] += int(argmax == gold)
                bucket["lat"].append(elapsed)
                bucket["covered"] += int(len(probs) == k)
            except Exception as error:  # noqa: BLE001
                print("ERROR", k, type(error).__name__, str(error)[:80], flush=True)
finally:
    client.close()
for k, b in sorted(stats.items()):
    n = b["n"]
    lat = sorted(b["lat"])
    print(
        f"K={k:3d} n={n:3d} gold_top={b['hit']}/{n}={b['hit']/n:.3f} "
        f"covered={b['covered']}/{n} lat_med={lat[len(lat)//2]:.2f}s lat_max={lat[-1]:.2f}s",
        flush=True,
    )
