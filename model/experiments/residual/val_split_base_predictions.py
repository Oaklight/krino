"""Per-item base-model predictions on the trainer's exact 20K validation split.

The residual arm's frozen baseline (cc 0.5113 overall; noul 0.4874, choice
0.5991, score 0.3083) is the argmax of the precomputed base distribution on the
split made by ``split_items(items, 0.2, 42)``, with the same ``_label_index``
rule the trainer uses.  No model forward is involved, so this file reproduces
the metrics per item rather than re-scoring the backbone.

Writes one JSON object per line:
    id, q_type, n_options, gold, pred, gold_index, pred_index, correct, probs

where ``gold`` and ``pred`` are the option texts and the indices are into the
option list, matching the gate's prediction records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from data.pipeline import load_jsonl
from model.scripts.train_residual import _label_index, _options, split_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items",
        type=Path,
        default=Path(
            "/home/pding/projects/krino-v3/data/benchmarks/residual_train_20000.jsonl"
        ),
    )
    parser.add_argument(
        "--base-probs",
        type=Path,
        default=Path(
            "/home/pding/projects/krino-v3/data/benchmarks/"
            "base_probs_20000_qwen35_4b.jsonl"
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/pding/projects/wt-233/scan/val_split_3999_base.pred.jsonl"),
    )
    args = parser.parse_args()

    base = {}
    with args.base_probs.open() as handle:
        for line in handle:
            record = json.loads(line)
            base[record["id"]] = torch.tensor(record["probs"], dtype=torch.float32)
    items = [item for item in load_jsonl(args.items) if item.id in base]
    train, val = split_items(items, args.val_fraction, args.split_seed)
    print(f"split train {len(train)} val {len(val)}", flush=True)

    rows = []
    per_type = defaultdict(list)
    for item in val:
        probs = base[item.id]
        target = _label_index(item)
        if target < 0:
            continue
        options = _options(item.question)
        pred = int(torch.argmax(probs).item())
        hit = pred == target
        k = probs.numel()
        per_type[item.question["type"]].append(
            (float(hit) - 1.0 / k) / (1.0 - 1.0 / k)
        )
        rows.append(
            {
                "id": item.id,
                "q_type": item.question["type"],
                "n_options": k,
                "gold": options[target] if target < len(options) else str(item.label),
                "pred": options[pred] if pred < len(options) else str(pred),
                "gold_index": target,
                "pred_index": pred,
                "correct": hit,
                "probs": [float(value) for value in probs],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )

    mean = lambda values: sum(values) / len(values)
    all_values = per_type["noul"] + per_type["choice"] + per_type["score"]
    trained = per_type["choice"] + per_type["score"]
    print(f"scored {len(all_values)}")
    print(f"frozen total cc {mean(all_values):.4f}")
    for q_type in ("noul", "choice", "score"):
        if per_type[q_type]:
            print(f"{q_type} {len(per_type[q_type])} {mean(per_type[q_type]):.4f}")
    print(f"frozen trained-type cc {mean(trained):.4f} n {len(trained)}")
    print(f"wrote {args.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
