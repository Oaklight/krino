import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if any(r.get("arm") == "residual_band20k_u50" for r in rows):
    raise SystemExit("already registered")
row = {
    "arm": "residual_band20k_u50",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
    "train": "data/benchmarks/band_up0.5.jsonl (8,968 items: drop base p(gold) > 0.5), same base probs cache, rank 32, KL 1.0, CE 0.2, seed 42, epochs 4, patience 1, max_length 2048",
    "gate": "promote if JevBench cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613; between the two the band branch closes as no-gain",
    "baselines": {"B2_5k_s42": 0.460317, "frozen_5k_mean": 0.4613, "D20K_20k": 0.460317},
    "status": "registered",
    "deviation": "Rufus-Air's upper cut is 0.8, but on this corpus p(gold) has p50 0.529 and the 0.8 cut keeps 80.4% (16,074 items), i.e. it would repeat the 20K arm that produced +0.000 cc. Using 0.5 (8,968 items) is the first cut that actually shifts the distribution toward the base's uncertain/wrong items. Band is computed from the cached base only, not re-scored with the current head each epoch; B2's alphas were small (choice -0.048), so drift is expected to be limited and is measured by band_selection --current-probs if a second pass is needed.",
    "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("registered", row["arm"], row["registered_at"])
