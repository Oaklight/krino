#!/usr/bin/env python3
"""Score the JevBench public tiers with a causal LM's own logit readout.

The trained heads are gated on JevBench (231 public items), but the untuned
bases have never been measured there, so the numbers cannot say whether the
task or the head is the problem. This script measures a base checkpoint with no
Krino head at all: the option text is scored as a continuation of
state + question, the option log-probs are softmaxed, and the accuracy rule and
tier aggregation come from ``model/evaluation/jevbench.py`` so the rows join the
table in ``research/13`` rather than starting a second one.

Usage:
    python model/scripts/eval_jevbench_logit.py \
        --model mistralai/Ministral-3-3B-Base-2512 --variant base \
        --output model/experiments/jevbench_logit_mistralai_Ministral-3-3B-Base-2512.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.jevbench import summarize  # noqa: E402
from model.evaluation.run_provenance import build_run_provenance  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402


def _record(scorer: LogitScorer, item: Any) -> dict[str, Any] | None:
    """Score one item and return a record in the shape ``summarize`` expects."""
    question = item.question
    q_type = question.get("type", "noul")
    state = item.state
    instructions = question.get("instructions", "")

    if q_type == "noul":
        context = f"State: {state}\nQuestion: {instructions}\nAnswer:"
        scores = scorer._score_options(context, [" yes", " no"])
        positive = item.label if isinstance(item.label, bool) else str(item.label).lower() == "true"
        return {
            "type": "noul",
            "logits": [scores[1], scores[0]],
            "label_order": [0, 1],
            "label_idx": 1 if positive else 0,
            "option_count": 2,
            "_item": item,
        }

    if q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        if item.label not in keys:
            return None
        descriptions = list(criteria.values())
        if scorer.strategy == "description":
            context = f"State: {state}\nQuestion: {instructions}\nAnswer:"
            option_texts = [f" {desc or key}" for key, desc in zip(keys, descriptions)]
        else:
            if len(keys) <= 10:
                labels = [str(i) for i in range(len(keys))]
            else:
                labels = [chr(65 + i) if i < 26 else f"opt{i}" for i in range(len(keys))]
            context = f"State: {state}\nQuestion: {instructions}\nOptions:\n"
            context += "\n".join(f"{labels[i]}. {descriptions[i] or keys[i]}" for i in range(len(keys)))
            context += "\nAnswer:"
            option_texts = [f" {label}" for label in labels]
        scores = scorer._score_options(context, option_texts)
        return {
            "type": "choice",
            "logits": scores,
            "label_order": list(range(len(keys))),
            "label_idx": keys.index(item.label),
            "option_count": len(keys),
            "_item": item,
        }

    if q_type == "score":
        criteria = question["criteria"]
        context = f"State: {state}\nQuestion: {instructions}\nLevels:\n"
        context += "\n".join(f"{i}. {desc}" for i, desc in enumerate(criteria))
        context += "\nLevel:"
        scores = scorer._score_options(context, [f" {i}" for i in range(len(criteria))])
        gold_idx = max(0, min(len(criteria) - 1, int(round(float(item.label)))))
        return {
            "type": "score",
            "logits": scores,
            "label_order": list(range(len(criteria))),
            "label_idx": gold_idx,
            "option_count": len(criteria),
            "_item": item,
        }

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=["base", "instruct"], default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--strategy", default="description", choices=["description", "label"])
    parser.add_argument("--norm", default="mean", choices=["mean", "sum"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    snapshot_path = args.snapshot
    if snapshot_path is None:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(args.model))

    print(f"Loading {args.model} from {snapshot_path}", flush=True)
    model, tokenizer = load_causal_lm(
        str(snapshot_path), device=args.device, dtype=dtype_map[args.dtype]
    )
    device = str(next(model.parameters()).device)
    scorer = LogitScorer(
        model, tokenizer, norm=args.norm, strategy=args.strategy, batch_size=64
    )

    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    print(f"Loaded {len(items)} JevBench items", flush=True)

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    failed = 0
    for index, item in enumerate(items):
        record = _record(scorer, item)
        if record is None:
            failed += 1
            continue
        records.append(record)
        predictions.append(
            {
                "id": item.id,
                "type": record["type"],
                "gold_idx": record["label_idx"],
                "option_count": record["option_count"],
                "logits": [round(value, 6) for value in record["logits"]],
                "correct": None,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    summary = summarize(records, failed=failed)
    positions = {(p["id"]): p for p in predictions}
    for record, prediction in zip(records, predictions):
        from model.evaluation.jevbench import record_correct

        prediction["correct"] = record_correct(record)
        assert positions[prediction["id"]] is prediction

    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy=args.strategy,
        norm=args.norm,
        root=ROOT,
        variant=args.variant,
        code_commit=args.code_commit,
    )
    payload = {
        "_provenance": provenance,
        "items": len(items),
        "n": summary["overall"]["total"],
        "failed": failed,
        "summary": summary,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    overall = summary["overall"]
    print(f"\noverall {overall['correct']}/{overall['total']} = {overall['accuracy']:.4f}")
    for tier, data in sorted(summary["by_tier"].items()):
        print(f"  {tier:<9} {data['correct']}/{data['total']} = {data['accuracy']:.4f}")
    for q_type, data in sorted(summary["by_type"].items()):
        print(f"  {q_type:<9} {data['correct']}/{data['total']} = {data['accuracy']:.4f}")
    if failed:
        print(f"unscored: {failed}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
