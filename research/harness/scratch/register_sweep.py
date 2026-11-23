import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("model/experiments/residual/manifest.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
have = {r.get("arm") for r in rows}
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
base = {
    "backbone": "Qwen/Qwen3-Reranker-4B",
    "readout": "marker, chunk 1, max_option_tokens 64, option_batch 16",
    "train": "data/benchmarks/residual_train_5000.jsonl + base_probs_5000.jsonl, seed 42, epochs 4, patience 1, max_length 2048, option_batch 16",
    "gate": "advance to confirm only if JevBench cc >= 0.4703 (B2 0.4603 + 1.0pp) and hard >= 0.459; an arm with cc < 0.4603 closes its knob; between is inconclusive. One knob changes per arm.",
    "status": "registered",
    "registered_at": now,
}
sweep = {
    "H_alpha_lr": {"alpha_lr": 0.03, "note": "default alpha-lr is 0.01; the gate is the only learned mechanism, so its step size is the first knob"},
    "H_ce": {"ce_weight": 0.5, "note": "default CE weight is 0.2 against KL 1.0"},
    "H_kl": {"kl_weight": 0.5, "note": "default KL weight is 1.0; weaker anchor lets the head move further from the base"},
    "H_rank": {"rank": 64, "note": "default rank is 32; doubles head capacity (251K -> ~500K params)"},
}
for arm, cfg in sweep.items():
    if arm in have:
        print("skip existing", arm)
        continue
    row = dict(base)
    row["arm"] = arm
    row["deviation"] = cfg["note"]
    row["knob"] = {k: v for k, v in cfg.items() if k != "note"}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("registered", arm)
