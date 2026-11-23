import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if any(r.get("arm") == "residual_D20K" for r in rows):
    raise SystemExit("residual_D20K already registered")
row = {
    "arm": "residual_D20K",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64",
    "train": "residual_train_20000.jsonl + base_probs_20000.jsonl (20,000 records, 0 missing), rank 32, KL 1.0, CE 0.2, seed 42, epochs 4, patience 1, max_length 2048",
    "gate": "promote if JevBench cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613; between the two the 4B data-scale branch closes as no-gain",
    "baselines": {
        "B2_s42": 0.460317,
        "B2_confirm_mean": 0.4613,
        "B2_confirm_best": 0.4690,
        "B2_hard": 0.459459,
    },
    "status": "registered",
    "deviation": "Only corpus size changes (5K -> 20K, stratified by source and type). Same backbone, readout, loss weights, seed and schedule as B2. Seed spread at 5K is 1.45 pp, so the promote line sits 1.1 pp above the best observed seed.",
    "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("registered", row["arm"], row["registered_at"])
