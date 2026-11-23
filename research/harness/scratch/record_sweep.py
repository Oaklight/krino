import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
results = {
    "H_alpha_lr": {"cc": 0.4369, "hard": 0.4414, "orig": 0.6806, "acc": 0.6320, "status": "killed", "verdict": "alpha-lr 0.03 loses 2.3 pp vs B2; alpha-lr knob closed"},
    "H_ce": {"cc": 0.4214, "hard": 0.4234, "orig": 0.6806, "acc": 0.6234, "status": "killed", "verdict": "CE 0.5 loses 3.9 pp; CE weight knob closed"},
    "H_kl": {"cc": 0.4747, "hard": 0.5045, "orig": 0.6667, "acc": 0.6580, "status": "advanced", "verdict": "KL 0.5 beats B2 by 1.44 pp and hard crosses 50; advances to confirm"},
    "H_rank": {"cc": 0.4517, "hard": 0.4685, "orig": 0.6667, "acc": 0.6407, "status": "killed", "verdict": "rank 64 loses 0.9 pp; rank knob closed"},
}
rows = []
for arm, r in results.items():
    rows.append({
        "arm": arm,
        "status": r["status"],
        "result": {"acc": r["acc"], "cc": r["cc"], "hard": r["hard"], "original": r["orig"], "easy": 1.0, "n": 231},
        "baseline": {"B2_cc": 0.460317, "B2_hard": 0.459459},
        "delta_cc_vs_B2": round(r["cc"] - 0.460317, 4),
        "gate": "advance if cc >= 0.4703 and hard >= 0.459; cc < 0.4603 closes the knob",
        "verdict": r["verdict"],
        "evidence": f"model/experiments/residual/{arm}/jevbench.json",
        "at": now,
    })
confirm = {
    "arm": "confirm_kl_5k",
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
    "train": "same 5K cache as B2, rank 32, KL weight 0.5, CE 0.2, seeds 43 and 44, epochs 4, patience 1, max_length 2048, option_batch 16",
    "gate": "both seeds cc >= 0.4703 and hard >= 0.459 -> confirmed; any seed cc <= 0.4613 -> rejected; otherwise inconclusive",
    "baselines": {"B2_s42_kl1.0": 0.460317, "H_kl_s42_kl0.5": 0.4747, "H_kl_hard": 0.5045},
    "status": "registered",
    "deviation": "Confirm stage for the one sweep knob that cleared the advance bar. KL weight 0.5 is the only change from B2.",
    "registered_at": now,
}
rows.append(confirm)
with path.open("a", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")
print("appended", len(rows))
