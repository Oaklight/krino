"""Summarize the nine-arm JevBench gates.

Reads each arm's per-item predictions, computes overall and hard-tier
chance-corrected accuracy and ECE, and records the seed/config for each arm.
"""

import os
import json
from pathlib import Path

import torch

ROOT = Path(os.path.expandvars("${KRINO_WORKTREE}"))
TAGS = [
    "lr1e-3_kl0", "lr1e-3_kl0.5", "lr1e-3_kl1.0",
    "lr3e-4_kl0", "lr3e-4_kl0.5", "lr3e-4_kl1.0",
    "lr1e-4_kl0", "lr1e-4_kl0.5", "lr1e-4_kl1.0",
]
CONTROL = {"accuracy": 0.6883, "corrected": 0.5187, "hard_corrected": 0.5045, "ece": 0.1175}


def ece(confidences, correct, n_bins=10):
    total = len(confidences)
    if not total:
        return 0.0
    error = 0.0
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        members = [
            i for i, conf in enumerate(confidences)
            if (low <= conf < high) or (index == n_bins - 1 and conf == high)
        ]
        if not members:
            continue
        avg_conf = sum(confidences[i] for i in members) / len(members)
        avg_acc = sum(1.0 if correct[i] else 0.0 for i in members) / len(members)
        error += abs(avg_acc - avg_conf) * len(members) / total
    return error


def metrics(rows):
    usable = [row for row in rows if row.get("n_options", 0) > 1]
    if not usable:
        return {"n": 0}
    correct = [bool(row["correct"]) for row in usable]
    corrected = [
        (1.0 if hit else 0.0) - 1.0 / row["n_options"]
        for hit, row in zip(correct, usable)
    ]
    corrected = [value / (1.0 - 1.0 / row["n_options"]) for value, row in zip(corrected, usable)]
    confidences = [max(row["probs"]) for row in usable]
    return {
        "n": len(usable),
        "accuracy": sum(correct) / len(correct),
        "corrected": sum(corrected) / len(corrected),
        "ece": ece(confidences, correct),
    }


def by_type(rows):
    out = {}
    for q_type in ("noul", "choice", "score"):
        subset = [row for row in rows if row.get("q_type") == q_type]
        if subset:
            out[q_type] = metrics(subset)
    return out


def main():
    report = {"control": CONTROL, "arms": {}}
    for tag in TAGS:
        path = ROOT / "scan" / tag / "jevbench.pred.jsonl"
        record = {"tag": tag, "preds": str(path)}
        if not path.exists():
            record["status"] = "missing"
            report["arms"][tag] = record
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        hard = [row for row in rows if str(row.get("id", "")).split("-", 1)[0] == "hard"]
        checkpoint = ROOT / "scan" / tag / "listwise.pt"
        config = {}
        if checkpoint.exists():
            config = torch.load(checkpoint, map_location="cpu", weights_only=False).get("config", {})
        record.update(
            {
                "status": "ok",
                "items": len(rows),
                "overall": metrics(rows),
                "hard": metrics(hard),
                "by_type": by_type(rows),
                "config": {
                    key: config.get(key)
                    for key in (
                        "lr", "alpha_lr", "lora_kl_weight", "lora_layers",
                        "lora_rank", "lora_alpha", "lora_targets",
                        "torch_seed", "seed", "split_seed",
                    )
                },
            }
        )
        report["arms"][tag] = record

    (ROOT / "scan" / "gate_report.json").write_text(json.dumps(report, indent=1))
    print("tag                 n   acc     cc      hard_cc  ece    lr      kl   torch_seed")
    for tag in TAGS:
        record = report["arms"][tag]
        if record.get("status") != "ok":
            print(f"{tag:18s} MISSING")
            continue
        overall = record["overall"]
        hard = record["hard"]
        config = record["config"]
        print(
            f"{tag:18s} {overall['n']:3d} {overall['accuracy']:.4f} {overall['corrected']:.4f} "
            f"{hard['corrected']:.4f}  {overall['ece']:.4f} {config['lr']} {config['lora_kl_weight']} "
            f"{config['torch_seed']}"
        )
    print("control            231 0.6883 0.5187 0.5045 0.1175")


if __name__ == "__main__":
    main()
