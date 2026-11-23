import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if any(r.get("arm") == "residual_teacher30k" for r in rows):
    raise SystemExit("already registered")
row = {
    "arm": "residual_teacher30k",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
    "train": "data/benchmarks/label_target_30k_jev.jsonl (30,000, all Jev-labeled) + base_probs_full.jsonl, rank 32, KL 1.0, CE 0.2, seed 42, epochs 4, patience 1, max_length 2048, option_batch 16",
    "gate": "promote if JevBench cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613; between the two the 30K recipe closes as no-gain",
    "baselines": {"D20K_hard_labels": 0.460317, "frozen_5k_mean": 0.4613},
    "status": "registered",
    "deviation": "Three changes at once versus D20K: Jev soft targets (100% coverage), band selection (base p(gold) <= 0.5), and a choice-heavy type mix (32/60/8). This is the full recipe, not an isolated teacher test; residual_teacher20k isolates the teacher effect on the same 20K split.",
    "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("registered", row["registered_at"])
