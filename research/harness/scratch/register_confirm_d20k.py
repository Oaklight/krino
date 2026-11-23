import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if any(r.get("arm") == "confirm_d20k" for r in rows):
    raise SystemExit("confirm_d20k already registered")
row = {
    "arm": "confirm_d20k",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64",
    "train": "same 20K cache and recipe as D20K; seeds 43 and 44; rank 32, KL 1.0, CE 0.2, epochs 4, patience 1, max_length 2048",
    "gate": "two seeds both >= 0.4459 -> 20K recipe stable (confirmed); any seed < 0.4395 -> rejected; otherwise inconclusive. Reported as the seed-spread band around D20K's primary cc",
    "baselines": {"D20K_s42_cc": None, "frozen_5k_mean": 0.4613, "raw_base": 0.4459},
    "status": "registered",
    "deviation": "Confirm stage for the data-scale arm, run in parallel with D20K's JevBench gate. Purpose is to measure seed spread at 4x data before interpreting a single-seed delta.",
    "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("registered", row["arm"], row["registered_at"])
