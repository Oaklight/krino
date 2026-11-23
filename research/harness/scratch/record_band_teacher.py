import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
have = {r.get("arm") for r in rows}
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
out = []
if "residual_band20k_u50" not in have or not any(
    r.get("arm") == "residual_band20k_u50" and r.get("status") == "killed" for r in rows
):
    out.append({
        "arm": "residual_band20k_u50",
        "status": "killed",
        "result": {"acc": 0.6407, "cc": 0.4549, "hard": 0.4685, "original": 0.6667, "easy": 1.0, "n": 231},
        "baseline": {"B2_cc": 0.460317, "frozen_5k_mean": 0.4613},
        "delta_cc_vs_B2": -0.0054,
        "verdict": "cc 0.4549 <= 0.4613 kill line; filtering to the base's uncertain items (p(gold) <= 0.5) does not help. Band-selection branch closed at this threshold.",
        "evidence": "model/experiments/residual/band20k_u50/jevbench.json",
        "at": now,
    })
if "residual_teacher20k" not in have:
    out.append({
        "arm": "residual_teacher20k",
        "backbone": "Qwen/Qwen3-Reranker-4B",
        "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
        "train": "data/benchmarks/residual_train_20000_jev.jsonl (19,350 Jev-labeled) + base_probs_20000.jsonl, rank 32, KL 1.0, CE 0.2, seed 42, epochs 4, patience 1, max_length 2048, option_batch 16",
        "gate": "promote if JevBench cc >= 0.4800 and hard >= 0.459; kill if cc <= 0.4613; between the two the teacher branch closes as no-gain",
        "baselines": {"D20K_hard_labels": 0.460317, "frozen_5k_mean": 0.4613},
        "status": "registered",
        "deviation": "KL stays at 1.0 so the only change against D20K is soft teacher targets (19,350/20,000 items; the rest fall back to hard labels). The KL 0.5 result is a separate confirm and is not combined here.",
        "registered_at": now,
    })
with path.open("a", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row) + "\n")
print("appended", len(out))
