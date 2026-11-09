#!/usr/bin/env python3
"""Per-request judgement: all K options in one sequence, K vocabulary reads (#324).

One sequence per item::

    State: ...
    Question: ...
    Option: {opt1}
    Does option 1 correctly answer the question? Answer:
    Option: {opt2}
    Does option 2 correctly answer the question? Answer:
    ...

Each ``Answer:`` is a read position.  The score is that position's
``log P(" yes") - log P(" no")`` through the model's own vocabulary head --
no trained parameters, no head.  With ``--direction causal`` an option's read
sees the options before it; with ``--direction full`` it sees all of them.

Usage:
    python model/scripts/eval_jevbench_judgement_per_request.py \\
        --model Qwen/Qwen3-Reranker-4B --direction causal \\
        --output model/experiments/logit_prov/jevbench_judgement_pr_causal.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.jevbench import record_correct, summarize  # noqa: E402
from model.evaluation.judgement_prompt import candidates, polarity_gold  # noqa: E402
from model.evaluation.run_provenance import build_run_provenance  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import full_attention_mask  # noqa: E402

ANSWER_TAIL = " Answer:"


def build_sequence(item: Any) -> tuple[str, list[int], int | None] | None:
    """Return the per-request text, each option's Answer-span end, and gold.

    The gold index is ``None`` for a criteria-less noul, where the single read
    answers the proposition itself rather than one of several options.
    """
    prepared = candidates(item)
    instructions = item.question.get("instructions", "")
    if prepared is None:
        gold = polarity_gold(item)
        if gold is None:
            return None
        text = f"State: {item.state}\nQuestion: {instructions}\nAnswer:"
        return text, [len(text)], None

    texts, gold = prepared
    parts = [f"State: {item.state}", f"Question: {instructions}"]
    ends: list[int] = []
    for index, option_text in enumerate(texts, start=1):
        parts.append(f"Option: {option_text}")
        parts.append(f"Does option {index} correctly answer the question?{ANSWER_TAIL}")
        ends.append(len("\n".join(parts)))
    return "\n".join(parts), ends, gold


def answer_token_indices(tokenizer: Any, text: str, char_ends: list[int]) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    indices = []
    for char_end in char_ends:
        target = char_end - 1
        found = None
        for index in range(len(offsets) - 1, -1, -1):
            start, end = offsets[index]
            if start <= target < end:
                found = index
                break
        if found is None:
            raise ValueError(f"no token covers character {target}")
        indices.append(found)
    return indices


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} tokenises to {ids}, not one id")
    return ids[0]


def yes_no_scores(lm_head: Any, read: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """``logit(yes) - logit(no)`` in float32 from a bf16 vocabulary head.

    Only the two embedding rows are upcast: converting the whole lm_head would
    also convert a tied embedding table and break the backbone's dtype.
    """
    weight = lm_head.weight
    direction = (weight[yes_id].float() - weight[no_id].float())
    return read.float() @ direction


def score_item(
    model: Any,
    tokenizer: Any,
    lm_head: Any,
    item: Any,
    *,
    direction: str,
    yes_id: int,
    no_id: int,
    device: str,
    use_cache: bool = False,
) -> tuple[dict[str, Any] | None, float]:
    built = build_sequence(item)
    if built is None:
        return None, 0.0
    text, char_ends, gold = built

    encoded = tokenizer(
        text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True
    )
    indices = answer_token_indices(tokenizer, text, char_ends)
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)

    tower = getattr(model, "model", model)
    mask: Any = attention
    if direction == "full":
        mask = {"full_attention": full_attention_mask(attention)}
    started = time.perf_counter()
    with torch.no_grad():
        output = tower(
            input_ids=input_ids,
            attention_mask=mask,
            use_cache=use_cache,
            output_hidden_states=False,
        )
        hidden = output.last_hidden_state[0]
        read = hidden[torch.tensor(indices, device=device)]
        scores = yes_no_scores(lm_head, read, yes_id, no_id).tolist()
    elapsed = time.perf_counter() - started

    if gold is None:
        delta = float(scores[0])
        positive = bool(
            polarity_gold(item)
        )
        return (
            {
                "type": item.question.get("type", "noul"),
                "logits": [delta],
                "label_order": [0],
                "label_idx": 0,
                "option_count": 1,
                "correct": (delta > 0) == positive,
                "mode": "polarity",
                "_item": item,
            },
            elapsed,
        )

    return (
        {
            "type": item.question.get("type", "noul"),
            "logits": [float(value) for value in scores],
            "label_order": list(range(len(scores))),
            "label_idx": gold,
            "option_count": len(scores),
            "mode": "per-request",
            "_item": item,
        },
        elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--direction", choices=["causal", "full"], required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--dataset",
        choices=["jevbench", "dev"],
        default="jevbench",
        help="jevbench: the 231-item public set; dev: the 828-row development set",
    )
    parser.add_argument("--dev-path", type=Path, default=ROOT / "data/benchmarks/dev/dev_800.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    yes_id = single_token_id(tokenizer, " yes")
    no_id = single_token_id(tokenizer, " no")
    lm_head = model.get_output_embeddings()

    if args.dataset == "dev":
        from model.evaluation.dev_set import load_dev_set

        items = list(load_dev_set(args.dev_path))
    else:
        items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    failed = 0
    for item in items:
        record, elapsed = score_item(
            model,
            tokenizer,
            lm_head,
            item,
            direction=args.direction,
            yes_id=yes_id,
            no_id=no_id,
            device=device,
        )
        if record is None:
            failed += 1
            continue
        record["correct"] = record_correct(record)
        records.append(record)
        latencies.append(elapsed)
        predictions.append(
            {
                "id": item.id,
                "type": record["type"],
                "gold_idx": record["label_idx"],
                "option_count": record["option_count"],
                "scores": [round(value, 6) for value in record["logits"]],
                "predicted_idx": max(range(len(record["logits"])), key=lambda i: record["logits"][i]),
                "mode": record["mode"],
                "latency_s": round(elapsed, 4),
                "correct": record["correct"],
            }
        )

    summary = summarize(records, failed=failed)
    latency = {
        "mean_s": statistics.fmean(latencies) if latencies else None,
        "median_s": statistics.median(latencies) if latencies else None,
        "n": len(latencies),
    }
    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot,
        dtype=args.dtype,
        device=device,
        strategy="judgement-per-request",
        norm="mean",
        root=ROOT,
        variant=None,
        code_commit=args.code_commit,
        extra={
            "readout": "per-request judgement: one sequence, K answer spans, each "
                       'scored log P(" yes") - log P(" no") through the vocabulary '
                       "head; no trained parameters",
            "readout_direction": args.direction,
            "read_position": "answer-span (K reads)",
            "head_kind": "vocabulary",
            "adaptation": "frozen",
            "yes_id": yes_id,
            "no_id": no_id,
            "dataset": args.dataset,
            "latency": latency,
        },
    )
    payload = {
        "_provenance": provenance,
        "items": len(items),
        "n": summary["overall"]["total"],
        "failed": failed,
        "summary": summary,
        "latency": latency,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {args.output}")
    print(json.dumps({"overall": summary["overall"], "latency": latency}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
