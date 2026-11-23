import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
rows = [
    {
        "arm": "residual_D20K",
        "status": "killed",
        "gate": "promote if cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613",
        "result": {
            "acc": 0.6450, "cc": 0.460317, "ece": 0.0806, "easy": 1.0,
            "hard": 0.477477, "original": 0.666667, "choice": 0.6043,
            "noul": 0.6757, "score": 0.8333, "qwk": 0.9226,
        },
        "baselines": {
            "B2_5k_s42": {"cc": 0.460317, "hard": 0.459459, "original": 0.708333},
            "frozen_5k_mean": 0.4613,
        },
        "delta_cc_vs_B2": 0.0,
        "alpha": {"noul": -0.0388, "choice": -0.2344, "score": -0.1622},
        "evidence": "model/experiments/residual/D20K/jevbench.json",
        "at": now,
    },
    {
        "arm": "confirm_d20k",
        "status": "stopped",
        "reason": "Primary arm residual_D20K killed by its pre-registered gate (cc 0.4603 <= 0.4613). Seeds 43/44 were confirm-stage only and were terminated after ~12 min to free GPUs.",
        "at": now,
    },
]
with path.open("a", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("appended", len(rows))
