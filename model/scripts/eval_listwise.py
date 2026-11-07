"""Score option sets with a trained listwise head.

The combined arm needs the listwise head to score a stage-1 shortlist rather
than the full set, so this CLI takes an optional subset per item and returns
predictions over whatever subset it was given.  With ``--base-probs`` the head
is fused exactly as in training::

    logits = log(p_base) + alpha_type * listwise_logits

Usage:
    python model/scripts/eval_listwise.py \
        --checkpoint model/experiments/listwise_b1/listwise.pt \
        --jevbench \
        --base-probs data/benchmarks/base_probs_jevbench_qwen35_4b.jsonl \
        --subsets data/benchmarks/stage1_top4.json \
        --output model/experiments/listwise_b1/hard.pred.jsonl
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
from model.src.listwise import ListwiseMarkerReadout
from model.src.lora import lora_kwargs_from_config


def _load_subsets(path: Path | None) -> dict[str, list[int]]:
    """``{item_id: [option indices]}``; empty means every option."""
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit("--subsets must be a JSON object of id -> index list")
    return {str(key): [int(index) for index in value] for key, value in data.items()}


def score_item(
    readout: ListwiseMarkerReadout,
    item,
    subset: list[int],
    base: torch.Tensor | None,
    alphas: dict[str, float],
    shifts: int,
    base_conditioned: bool = False,
    head_only: bool = False,
) -> tuple[int, list[float]]:
    """Predicted option index and per-option probabilities over ``subset``."""
    q = item.question
    options = _options(q)
    q_type = q["type"]
    item_shifts = shifts if q_type == "choice" else 1
    picked = None
    if base is not None:
        picked = base[torch.tensor(subset, device=base.device)].clamp_min(1e-6)
        picked = picked / picked.sum()
    probs = readout.score_shifted(
        item.state,
        q.get("instructions", ""),
        [options[index] for index in subset],
        shifts=item_shifts,
        base_probs=picked if base_conditioned else None,
    ).reshape(-1)
    if picked is not None and not head_only:
        alpha = alphas.get(q_type, 0.0)
        probs = torch.softmax(
            torch.log(picked.to(probs.device)) + alpha * torch.log(probs), dim=-1
        )
    local = int(torch.argmax(probs).item())
    return subset[local], probs.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--items", type=Path, help="TypedQuestion JSONL")
    source.add_argument(
        "--jevbench",
        action="store_true",
        help="Score the JevBench tiers through load_jevbench",
    )
    parser.add_argument("--base-probs", type=Path, default=None)
    parser.add_argument(
        "--subsets",
        type=Path,
        default=None,
        help="JSON {item_id: [option indices]}; omitted means the full set",
    )
    parser.add_argument(
        "--base-conditioned",
        action="store_true",
        help="Pass the base distribution to the head as its two per-option "
        "features; must match the checkpoint",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Keep the base features but skip the alpha fusion, so the score "
        "is the head's alone",
    )
    parser.add_argument("--shifts", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("readout") != "listwise":
        raise SystemExit(
            f"{args.checkpoint} is not a listwise checkpoint "
            f"(readout={config.get('readout')!r})"
        )
    alphas = checkpoint["alpha"]
    shifts = args.shifts if args.shifts is not None else int(config.get("eval_shifts", 4))
    trained_conditioned = bool(config.get("base_conditioned", False))
    if args.base_conditioned != trained_conditioned:
        raise SystemExit(
            f"--base-conditioned={args.base_conditioned} does not match the "
            f"checkpoint (trained with base_conditioned={trained_conditioned})"
        )
    if args.base_conditioned and args.base_probs is None:
        raise SystemExit("--base-conditioned needs --base-probs")

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
    readout = ListwiseMarkerReadout(model)

    base_probs: dict[str, torch.Tensor] = {}
    if args.base_probs is not None:
        for line in args.base_probs.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                base_probs[record["id"]] = torch.tensor(
                    record["probs"], dtype=torch.float32, device=args.device
                )
    subsets = _load_subsets(args.subsets)

    # JevBench rows carry ``expected``/``labels`` rather than a TypedQuestion
    # question dict, so they go through the same loader the gate uses.
    items = list(load_jevbench()) if args.jevbench else load_jsonl(args.items)
    if args.limit:
        items = items[: args.limit]
    records = []
    hits = 0
    corrected = 0.0
    scored = 0
    for item in items:
        q = item.question
        options = _options(q)
        subset = subsets.get(item.id, list(range(len(options))))
        if not subset:
            continue
        base = base_probs.get(item.id)
        with torch.no_grad():
            pred, probs = score_item(
                readout,
                item,
                subset,
                base,
                alphas,
                shifts,
                args.base_conditioned,
                args.head_only,
            )
        gold = _label_index(item)
        hit = gold >= 0 and pred == gold
        k = len(subset)
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
                "probs": probs,
                "subset": subset,
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
                "shifts": shifts,
                "base_conditioned": args.base_conditioned,
                "head_only": args.head_only,
                "subset_source": str(args.subsets) if args.subsets else "full",
            },
            indent=1,
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
