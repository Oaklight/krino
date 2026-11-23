import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if any(r.get("arm") == "confirm_teacher20k" for r in rows):
    raise SystemExit("already registered")
row = {
    "arm": "confirm_teacher20k",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
    "train": "same 20K Jev-labeled file and recipe as residual_teacher20k; seeds 43 and 44; rank 32, KL 1.0, CE 0.2, epochs 4, patience 1, max_length 2048, option_batch 16",
    "gate": "both seeds cc >= 0.4800 and hard >= 0.459 -> confirmed; any seed cc <= 0.4613 -> rejected; otherwise inconclusive",
    "baselines": {"D20K_hard_labels": 0.460317, "frozen_5k_mean": 0.4613},
    "status": "registered",
    "deviation": "Confirm seeds launched BEFORE the seed-42 primary gate is read, because the KL 0.5 sweep produced a single-seed false positive (0.4747 on seed 42, 0.4538/0.4545 on confirm) that cost a full confirm cycle. GPUs 0/1 were idle and each arm is ~40 min, so measuring the seed spread in parallel is cheaper than discovering it afterwards.",
    "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("registered", row["registered_at"])
