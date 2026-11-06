"""Score JevBench (or any TypedQuestion set) with a two-stage pairwise arm.

Stage 1 takes the top-m options of the base distribution; stage 2 duels them
pairwise and fuses::

    logits = log(p_base) + alpha_type * pairwise_log_probs

Options outside the shortlist get probability 0, so a gold that stage 1 drops
counts as wrong, which is the honest reading of a two-stage arm.

Two files are written: the per-item JSONL (``--output``) and the
``{"predictions": [...]}`` wrapper (``--result``) that
``model/scripts/paired_mcnemar.py`` reads.

Usage:
    python model/scripts/eval_pairwise.py \
        --checkpoint model/experiments/pairwise_D20K_qwen35/pairwise.pt \
        --jevbench \
        --base-probs data/benchmarks/base_probs_jevbench_qwen35_4b.jsonl \
        --output results/gate/pairwise/jevbench.pred.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from data.pipeline import load_jevbench, load_jsonl
from model.scripts.train_residual import _label_index, _options
from model.src.backbone import load_causal_lm
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig
from model.src.lora import lora_kwargs_from_config
from model.src.pairwise import PairwiseMarkerReadout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--items", type=Path, help="TypedQuestion JSONL")
    source.add_argument(
        "--jevbench",
        action="store_true",
        help="Score the JevBench tiers through load_jevbench",
    )
    parser.add_argument(
        "--base-probs",
        type=Path,
        required=True,
        help="Stage-1 distribution from the raw backbone",
    )
    parser.add_argument("--m", type=int, default=None, help="Shortlist size")
    parser.add_argument("--output", type=Path, default=None, help="Per-item JSONL")
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help='paired_mcnemar wrapper; defaults to --output with .json',
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("readout") != "pairwise":
        raise SystemExit(
            f"{args.checkpoint} is not a pairwise checkpoint "
            f"(readout={config.get('readout')!r})"
        )
    alphas = checkpoint["alpha"]
    m = args.m if args.m is not None else int(config.get("shortlist_m", 3))

    backbone, tokenizer = load_causal_lm(
        config["backbone"], device=args.device, freeze=True, flash_attention=False
    )
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=True,
        max_length=config["max_length"],
        gradient_checkpointing=False,
    )
    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=config["rank"],
        mlp_layers=0,
        gpu_config=gpu,
        max_length=config["max_length"],
        option_marker=True,
        option_chunk_size=1,
        option_max_tokens=int(config.get("option_max_tokens", 64)),
        **lora_kwargs_from_config(config),
    )
    heads = checkpoint["heads"]
    missing, unexpected = model.load_state_dict(heads, strict=False)
    missing = [name for name in missing if not name.startswith("backbone.")]
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not fit the model: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    model.eval()
    readout = PairwiseMarkerReadout(model)

    base_probs: dict[str, torch.Tensor] = {}
    for line in args.base_probs.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            base_probs[record["id"]] = torch.tensor(
                record["probs"], dtype=torch.float32, device=args.device
            )

    # JevBench rows carry expected/labels rather than a TypedQuestion question
    # dict, so they go through the same loader the gate uses.
    items = list(load_jevbench()) if args.jevbench else load_jsonl(args.items)
    if args.limit:
        items = items[: args.limit]

    records = []
    hits = 0
    corrected = 0.0
    scored = 0
    missed = 0
    for item in items:
        q = item.question
        options = _options(q)
        base = base_probs.get(item.id)
        if base is None:
            continue
        gold = _label_index(item)
        if q["type"] == "noul":
            probs = base
            shortlist = list(range(len(options)))
        else:
            with torch.no_grad():
                probs, shortlist = readout.shortlist_probabilities(
                    item.state,
                    q.get("instructions", ""),
                    options,
                    base,
                    float(alphas.get(q["type"], 0.0)),
                    m=m,
                    order_sensitive=q["type"] == "choice",
                )
            if gold not in shortlist:
                missed += 1
        pred = int(torch.argmax(probs).item())
        hit = gold >= 0 and pred == gold
        # Chance is over the full option set: the arm still had K candidates.
        k = len(options)
        chance = 1.0 / k
        hits += int(hit)
        corrected += (float(hit) - chance) / (1.0 - chance)
        scored += 1
        records.append(
            {
                "id": item.id,
                "q_type": q["type"],
                "n_options": k,
                "gold": options[gold] if gold >= 0 else None,
                "pred": options[pred],
                "correct": bool(hit),
                "probs": probs.tolist(),
                "shortlist": shortlist,
            }
        )

    accuracy = hits / scored if scored else 0.0
    cc = corrected / scored if scored else 0.0
    print(
        json.dumps(
            {
                "items": scored,
                "accuracy": accuracy,
                "corrected": cc,
                "m": m,
                "missed_shortlist": missed,
            },
            indent=1,
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )
        result_path = args.result or args.output.with_suffix(".json")
        result_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {"id": row["id"], "correct": row["correct"]}
                        for row in records
                    ],
                    "m": m,
                    "accuracy": accuracy,
                    "corrected": cc,
                    "missed_shortlist": missed,
                }
            )
        )
        print(f"wrote {args.output} and {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
