"""Probe the real Jev API on high-K choice questions."""
import random
import time
from pathlib import Path

from jev_client import JevClient

from data.pipeline import load_jsonl

items = [
    i for i in load_jsonl(Path("data/benchmarks/corpus_full.jsonl"))
    if i.question["type"] == "choice" and len(i.question["criteria"]) == 80
][:5]
print("items", len(items))
rng = random.Random(42)
client = JevClient()
try:
    for item in items:
        crit = item.question["criteria"]
        gold = str(item.label)
        if gold not in crit:
            print("skip", item.id, "gold not in criteria")
            continue
        others = [k for k in crit if k != gold]
        for k in (2, 5, 20, 50, 80):
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
                print(
                    f"K={k:3d} lat={elapsed:6.2f}s n={len(probs):3d} "
                    f"sum={sum(probs.values()):.3f} gold_top={argmax == gold} "
                    f"conf={ans.get('confidence')}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - report and continue
                elapsed = time.monotonic() - start
                print(f"K={k:3d} ERROR lat={elapsed:6.2f}s {type(error).__name__}: {str(error)[:120]}", flush=True)
finally:
    client.close()
