#!/usr/bin/env python3
"""Score JevBench public with a per-option judgement readout.

The continuation readout asks "how likely is this text to follow ``Answer:``",
which is a generation question. Qwen3-Reranker's native readout asks whether a
document meets the requirements, which is a judgement question, and the two
disagree by 14.8 points on the same checkpoint (#277). This script asks the
judgement question in a form a base model has seen: the state and question are
stated once, then one option at a time is presented and the model answers a
direct yes/no question about it.

For every option the prompt is::

    State: {state}
    Question: {instructions}
    Option: {option}
    Does this option correctly answer the question? Answer:

and the score is ``log P(" yes")`` against ``log P(" no")`` at the final
position. Higher is better; the options are ranked independently, as the native
reranker ranks them, so the values do not sum to one. The exact template is
recorded in ``_provenance.prompt_template``, one template for all items.

The accuracy rule and the tier aggregation come from
``model/evaluation/jevbench.py``, so a row here joins the table in
``research/13`` rather than starting a second one.

Usage:
    python model/scripts/eval_jevbench_judgement.py \\
        --model Qwen/Qwen3-4B-Base --variant base \\
        --output model/experiments/logit_prov/jevbench_judgement_Qwen_Qwen3-4B-Base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.dev_set import load_dev_set  # noqa: E402
from model.evaluation.jevbench import summarize  # noqa: E402
from model.evaluation.judgement_prompt import (  # noqa: E402
    JUDGEMENT_TEMPLATE,
    NO,
    YES,
    candidates,
    render,
)
from model.evaluation.run_provenance import build_run_provenance  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402


def score_item(scorer: LogitScorer, item: Any) -> dict[str, Any] | None:
    """Rank one item's options by the judgement readout."""
    prepared = candidates(item)
    if prepared is None:
        return None
    texts, gold = prepared
    instructions = item.question.get("instructions", "")

    logp_yes = []
    for text in texts:
        scores = scorer.score_options(
            render(item.state, instructions, text), [YES, NO]
        )
        logp_yes.append(scores[0])

    return {
        "type": item.question.get("type", "noul"),
        "logits": logp_yes,
        "label_order": list(range(len(texts))),
        "label_idx": gold,
        "option_count": len(texts),
        "_item": item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=["base", "instruct"], default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dataset", choices=["jevbench", "dev"], default="jevbench",
                        help="jevbench: the 231-item public set (acceptance); dev: the 800-item development set (#298)")
    parser.add_argument("--dev-path", type=Path, default=ROOT / "data/benchmarks/dev/dev_800.jsonl")
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
    scorer = LogitScorer(model, tokenizer, norm="mean", strategy="description", batch_size=1)

    if args.dataset == "dev":
        items = list(load_dev_set(args.dev_path))
    else:
        items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    print(f"Loaded {len(items)} JevBench items", flush=True)

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    failed = 0
    started = time.monotonic()
    for index, item in enumerate(items):
        record = score_item(scorer, item)
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
                "logp_yes": [round(value, 6) for value in record["logits"]],
                # Written for every type, including noul, so a consumer that
                # reads predicted_idx never has to fall back to the score array.
                "predicted_idx": record["logits"].index(max(record["logits"])),
                "correct": None,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    from model.evaluation.jevbench import record_correct

    for record, prediction in zip(records, predictions):
        prediction["correct"] = record_correct(record)

    summary = summarize(records, failed=failed)
    elapsed = time.monotonic() - started

    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy="judgement",
        norm="mean",
        root=ROOT,
        variant=args.variant,
        code_commit=args.code_commit,
        extra={
            "readout": 'per-option judgement, log P(" yes") against log P(" no") at "Answer:"',
            "prompt_template": JUDGEMENT_TEMPLATE,
            "option_texts": "noul: question criteria, true first; choice: description; "
                            "score: 'Level {i}: {description}'",
            "elapsed_s": round(elapsed, 1),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
