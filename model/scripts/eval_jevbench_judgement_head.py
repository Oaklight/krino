#!/usr/bin/env python3
"""Score JevBench public with a trained two-logit judgement head (#324).

The head is the arm's only trained component; the backbone is frozen.  The
prompt builder is the same one the cache was built with
(``judgement_objective.build_prompts``), so the arm cannot drift from the
objective.

Arms differ in three recorded fields: ``readout_direction`` (causal, full),
``read_position`` (answer, option) and ``adaptation`` (frozen, lora).  This
script covers the frozen arms; LoRA arms use the same tail with an adapted
backbone.

Usage:
    python model/scripts/eval_jevbench_judgement_head.py \\
        --model Qwen/Qwen3-Reranker-4B --head runs/arm_b/head.pt \\
        --direction causal --position answer \\
        --output model/experiments/logit_prov/jevbench_judgement_head_arm_b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.jevbench import record_correct, summarize  # noqa: E402
from model.evaluation.judgement_prompt import polarity_gold  # noqa: E402
from model.evaluation.run_provenance import build_run_provenance  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import (  # noqa: E402
    TwoWayHead,
    build_prompts,
    read_token_index,
    run_backbone,
)


def load_head(path: Path, device: str) -> tuple[TwoWayHead, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = TwoWayHead(payload["hidden_size"], yes_bias=payload.get("yes_bias", 2.0))
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()
    return head, payload


def score_item(
    model: Any,
    tokenizer: Any,
    head: TwoWayHead,
    item: Any,
    *,
    direction: str,
    position: str,
    device: str,
) -> dict[str, Any] | None:
    built = build_prompts(item, row_id=item.id, source="jevbench")
    if not built:
        return None

    texts = [prompt.text for prompt in built]
    encoded = [tokenizer(text, add_special_tokens=False) for text in texts]
    width = max(len(row["input_ids"]) for row in encoded)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    read_index = []
    for row, (prompt, enc) in enumerate(zip(built, encoded)):
        ids = enc["input_ids"]
        input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
        attention[row, : len(ids)] = 1
        if position == "option" and prompt.has_option_span:
            read_index.append(read_token_index(tokenizer, prompt.text, prompt.option_char_end))
        else:
            read_index.append(len(ids) - 1)

    with torch.no_grad():
        hidden = run_backbone(
            model, input_ids, attention, bidirectional=direction == "full"
        )
        rows = torch.arange(len(built), device=device)
        index = torch.tensor(read_index, device=device)
        logits = head(hidden[rows, index].float())

    scores = TwoWayHead.score(logits).tolist()

    if len(built) == 1 and not built[0].has_option_span:
        gold = polarity_gold(item)
        if gold is None:
            return None
        delta = float(scores[0])
        return {
            "type": item.question.get("type", "noul"),
            "logits": [delta],
            "label_order": [0],
            "label_idx": 0,
            "option_count": 1,
            "correct": (delta > 0) == bool(gold),
            "mode": "polarity",
            "_item": item,
        }

    gold_index = next(i for i, prompt in enumerate(built) if prompt.label == 1)
    return {
        "type": item.question.get("type", "noul"),
        "logits": scores,
        "label_order": list(range(len(scores))),
        "label_idx": gold_index,
        "option_count": len(scores),
        "mode": "option",
        "_item": item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--direction", choices=["causal", "full"], required=True)
    parser.add_argument("--position", choices=["answer", "option"], required=True)
    parser.add_argument("--adaptation", choices=["frozen", "lora"], default="frozen")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.direction == "full" and args.position == "answer":
        raise SystemExit("a full mask adds nothing at the answer position")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    snapshot = args.snapshot
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(args.model))
    model, tokenizer = load_causal_lm(str(snapshot), device=args.device, dtype=dtype_map[args.dtype])
    device = str(next(model.parameters()).device)
    head, head_payload = load_head(args.head, device)

    items = list(load_jevbench())
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    failed = 0
    for item in items:
        record = score_item(
            model,
            tokenizer,
            head,
            item,
            direction=args.direction,
            position=args.position,
            device=device,
        )
        if record is None:
            failed += 1
            continue
        record["correct"] = record_correct(record)
        records.append(record)
        predictions.append(
            {
                "id": item.id,
                "type": record["type"],
                "gold_idx": record["label_idx"],
                "option_count": record["option_count"],
                "scores": [round(value, 6) for value in record["logits"]],
                "predicted_idx": max(range(len(record["logits"])), key=lambda i: record["logits"][i]),
                "mode": record["mode"],
                "correct": record["correct"],
            }
        )

    summary = summarize(records, failed=failed)
    head_sha256 = hashlib.sha256(args.head.read_bytes()).hexdigest()
    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot,
        dtype=args.dtype,
        device=device,
        strategy="judgement-head",
        norm="none",
        root=ROOT,
        variant=None,
        code_commit=args.code_commit,
        extra={
            "readout": "two-logit yes/no head over one read-position hidden state; "
                       "score = logit(yes) - logit(no); no base-logit fusion",
            "readout_direction": args.direction,
            "read_position": args.position,
            "head_kind": "linear-2logit",
            "adaptation": args.adaptation,
            "head_sha256": head_sha256,
            "head_training": {
                "state_key": head_payload.get("state_key"),
                "best_epoch": head_payload.get("best_epoch"),
                "epochs_run": head_payload.get("epochs_run"),
                "train_rows": head_payload.get("train_rows"),
                "eval_rows": head_payload.get("eval_rows"),
                "train_shards_sha256": head_payload.get("train_shards_sha256"),
                "yes_bias": head_payload.get("yes_bias"),
            },
        },
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
    print(f"written {args.output}")
    print(json.dumps(summary["overall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
