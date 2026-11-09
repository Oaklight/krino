#!/usr/bin/env python3
"""Precompute read-position hidden states for the judgement objective (#324).

The backbone is frozen in every arm this feeds, so the head can be trained on
cached states without a forward pass per epoch.  One causal pass yields both
causal read positions (``answer`` and ``option``); a full-attention pass yields
the bidirectional ``option`` state.

Usage:
    python model/scripts/precompute_judgement_states.py \\
        --plan data/benchmarks/judgement_plan.jsonl \\
        --model Qwen/Qwen3-Reranker-4B --direction causal --position both \\
        --output data/benchmarks/judgement_states/causal --shard 0/8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import read_token_index, run_backbone  # noqa: E402


def load_plan(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--direction", choices=["causal", "full"], required=True)
    parser.add_argument("--position", choices=["answer", "option", "both"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="shard file stem")
    parser.add_argument("--shard", default="0/1", help="i/n, modulo over plan lines")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    if args.direction == "full" and args.position == "answer":
        raise SystemExit("a full mask adds nothing at the answer position; use --position option")

    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    plan = load_plan(args.plan)
    lines = [entry for index, entry in enumerate(plan) if index % shard_count == shard_index]

    snapshot = args.snapshot
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(args.model))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_causal_lm(str(snapshot), device=None, dtype=dtype)
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    device = str(next(model.parameters()).device)

    encoded: list[dict] = []
    skipped_long = 0
    for line_no, entry in enumerate(lines):
        ids = tokenizer(entry["text"], add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_length:
            skipped_long += 1
            continue
        option_end = entry["option_char_end"]
        answer_index = len(ids) - 1
        option_index = (
            read_token_index(tokenizer, entry["text"], option_end)
            if option_end is not None
            else answer_index
        )
        encoded.append(
            {
                "line": line_no,
                "ids": ids,
                "answer_index": answer_index,
                "option_index": option_index,
                "label": int(entry["label"]),
                "row_id": entry["row_id"],
                "option_index_field": int(entry["option_index"]),
                "n_options": int(entry["n_options"]),
                "q_type": entry["q_type"],
                "split": entry["split"],
                "source": entry["source"],
            }
        )

    # Bucket by length so padding waste stays small.
    encoded.sort(key=lambda item: len(item["ids"]))
    collected: dict[str, list] = {"answer": [], "option": []}
    meta: list[dict] = []
    started = time.monotonic()
    done = 0
    for start in range(0, len(encoded), args.batch_size):
        batch = encoded[start : start + args.batch_size]
        width = max(len(item["ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), width), pad_id, dtype=torch.long, device=device
        )
        attention = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            length = len(item["ids"])
            input_ids[row, :length] = torch.tensor(item["ids"], device=device)
            attention[row, :length] = 1
        with torch.no_grad():
            hidden = run_backbone(
                model,
                input_ids,
                attention,
                bidirectional=args.direction == "full",
            )
        rows = torch.arange(len(batch), device=device)
        if args.position in ("answer", "both"):
            index = torch.tensor([item["answer_index"] for item in batch], device=device)
            collected["answer"].append(hidden[rows, index].to(torch.float16).cpu().numpy())
        if args.position in ("option", "both"):
            index = torch.tensor([item["option_index"] for item in batch], device=device)
            collected["option"].append(hidden[rows, index].to(torch.float16).cpu().numpy())
        meta.extend(batch)
        done += len(batch)
        if done % 4096 < args.batch_size:
            rate = done / max(time.monotonic() - started, 1e-9)
            print(f"  {done:,}/{len(encoded):,}  {rate:.1f}/s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": np.array([item["label"] for item in meta], dtype=np.int8),
        "row_ids": np.array([item["row_id"] for item in meta]),
        "option_index": np.array([item["option_index_field"] for item in meta], dtype=np.int16),
        "n_options": np.array([item["n_options"] for item in meta], dtype=np.int16),
        "q_type": np.array([item["q_type"] for item in meta]),
        "split": np.array([item["split"] for item in meta]),
        "source": np.array([item["source"] for item in meta]),
        "direction": np.array(args.direction),
        "skipped_long": np.array(skipped_long),
    }
    for key, arrays in collected.items():
        if arrays:
            payload[f"h_{key}"] = np.concatenate(arrays, axis=0)
    output = args.output.with_name(f"{args.output.name}_shard{shard_index}of{shard_count}.npz")
    np.savez(output, **payload)
    print(
        f"wrote {output}  rows {len(meta):,}  skipped_long {skipped_long:,}  "
        f"elapsed {time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
