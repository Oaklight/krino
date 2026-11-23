import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
row = {
    "arm": "residual_teacher20k",
    "status": "promoted",
    "gate": "promote if JevBench cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613",
    "result": {"acc": 0.6667, "cc": 0.4892, "hard": 0.4865, "original": 0.7222, "easy": 1.0, "n": 231},
    "baseline": {"B2_5k_s42": 0.460317, "frozen_5k_mean": 0.4613, "B2_hard": 0.459459, "B2_original": 0.708333},
    "delta_cc_vs_B2": 0.0289,
    "delta_cc_vs_frozen_mean": 0.0279,
    "evidence": "model/experiments/residual/teacher20k/jevbench.json",
    "caveat": "Single seed. KL 0.5 showed a 1.4 pp seed-42 false positive earlier today, so this needs confirm_teacher20k (seeds 43/44, alreading running) before it is treated as confirmed.",
    "at": now,
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("recorded")
