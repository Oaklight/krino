#!/usr/bin/env python3
"""Split a residual arm's JevBench accuracy into base, head-alone and fused.

For every item the fused distribution is ``log p_base + alpha_type * head``.
This script reports, per tier and per question type, how often the base, the
head on its own and the fusion pick the gold option, how many base errors the
fusion and the head fix, and an alpha sweep per type with the accuracy at the
trained alpha computed exactly (not read off the grid).

Usage::

    python model/scripts/head_decomposition.py \\
        --residual-heads model/experiments/residual/D20K/residual.pt \\
        --output model/experiments/residual/D20K/head_decomposition.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from data.pipeline import load_jevbench
from model.scripts.eval_gate import ResidualScorer, _raw_options
from model.scripts.train_residual import _label_index, head_logits

ALPHA_GRID = (-1.0, -0.75, -0.5, -0.35, -0.25, -0.15, -0.1, 0.0, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0)


def _accuracy(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else float("nan")


def _group(rows: list[dict]) -> dict:
    wrong = [row for row in rows if not row["base"]]
    return {
        "n": len(rows),
        "base": _accuracy(rows, "base"),
        "head": _accuracy(rows, "head"),
        "fused": _accuracy(rows, "fused"),
        "base_wrong": len(wrong),
        "fused_fixes": sum(row["fused"] for row in wrong),
        "head_fixes": sum(row["head"] for row in wrong),
        "fused_breaks": sum(row["base"] and not row["fused"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--residual-heads", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scorer = ResidualScorer(args.model, args.device, args.residual_heads, warmup=True)
    alpha = {key: float(value) for key, value in scorer.alpha.items()}
    rows: list[dict] = []
    tensors: list[tuple[str, torch.Tensor, torch.Tensor, int]] = []
    for item in load_jevbench():
        question = item.question
        scores = scorer.base.score_many(
            item.state, question.get("instructions", ""), question["type"], _raw_options(question)
        )
        total = sum(scores) or 1.0
        base = torch.tensor([s / total for s in scores], dtype=torch.float32)
        head = head_logits(scorer.model, item, question).reshape(-1).float().cpu()
        target = _label_index(item)
        if target < 0 or target >= len(base):
            continue
        q_type = question["type"]
        fused = torch.log(base.clamp_min(1e-6)) + alpha.get(q_type, 0.0) * head
        top2 = torch.sort(base, descending=True).values[:2]
        rows.append(
            {
                "id": item.id,
                "tier": item.id.split("-", 1)[0],
                "type": q_type,
                "base": int(base.argmax()) == target,
                "head": int(head.argmax()) == target,
                "fused": int(fused.argmax()) == target,
                "base_log_margin": float(torch.log(top2[0]) - torch.log(top2[1])),
            }
        )
        tensors.append((q_type, base, head, target))

    by_tier: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
        by_type[row["type"]].append(row)

    sweep = {}
    for q_type in sorted(by_type):
        subset = [t for t in tensors if t[0] == q_type]

        def acc_at(a: float) -> float:
            hits = sum(
                int((torch.log(b.clamp_min(1e-6)) + a * h).argmax()) == y for _, b, h, y in subset
            )
            return hits / len(subset)

        curve = {f"{a:+.2f}": acc_at(a) for a in ALPHA_GRID}
        best = max(curve.items(), key=lambda kv: kv[1])
        sweep[q_type] = {
            "n": len(subset),
            "trained_alpha": alpha.get(q_type, 0.0),
            "acc_at_trained_alpha": acc_at(alpha.get(q_type, 0.0)),
            "acc_at_zero": curve["+0.00"],
            "best_alpha": float(best[0]),
            "best_acc": best[1],
            "curve": curve,
        }

    margins = sorted(row["base_log_margin"] for row in rows)
    result = {
        "residual_heads": str(args.residual_heads),
        "backbone": args.model,
        "alpha": alpha,
        "overall": _group(rows),
        "base_log_margin_p50": margins[len(margins) // 2] if margins else None,
        "by_tier": {k: _group(v) for k, v in sorted(by_tier.items())},
        "by_type": {k: _group(v) for k, v in sorted(by_type.items())},
        "alpha_sweep": sweep,
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("alpha", "overall", "by_type")}, indent=2))
    for q_type, entry in sweep.items():
        print(q_type, {k: v for k, v in entry.items() if k != "curve"})


if __name__ == "__main__":
    main()
