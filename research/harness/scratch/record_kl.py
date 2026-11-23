import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
row = {
    "arm": "confirm_kl_5k",
    "status": "rejected",
    "gate": "both seeds cc >= 0.4703 and hard >= 0.459 -> confirmed; any seed cc <= 0.4613 -> rejected",
    "results": {
        "s43": {"acc": 0.6450, "cc": 0.4538, "hard": 0.4505, "original": 0.7083, "easy": 1.0},
        "s44": {"acc": 0.6450, "cc": 0.4545, "hard": 0.4505, "original": 0.7083, "easy": 1.0},
        "s42": {"cc": 0.4747, "hard": 0.5045},
    },
    "baseline": {"B2_cc": 0.460317, "frozen_5k_mean": 0.4613},
    "verdict": "Both confirm seeds land at 0.4538/0.4545, below the 0.4613 reject line. The seed-42 KL 0.5 result (0.4747) does not replicate; the KL knob is closed. All four sweep knobs are now closed and the frozen marker-residual line stays at the 0.4613 mean.",
    "evidence": "model/experiments/residual/{KL43,KL44}/jevbench.json",
    "at": now,
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("recorded")
