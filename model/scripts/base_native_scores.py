#!/usr/bin/env python3
"""Score the short-list choice items with the *base* reranker, for a three-way blend.

The blend in ``research/19`` mixes the trained head with the fine-tuned
backbone's own readout, and it works because the two fail on different items.
But the per-source table there shows something else: the trained head is far
worse than the fine-tuned readout on ARC, race, authored144, CommonsenseQA and
MMLU. If fine-tuning cost the backbone reranking ability, then the *original*
reranker -- never trained on our task -- may be a better second opinion on
exactly those items than the fine-tuned one.

This produces the third score vector so that question can be asked offline. It
reproduces ``blend_scorers.py``'s item order exactly -- same seed, same
per-source sampling, same config order -- so the output rows line up with that
run's ``--dump-records`` output positionally. Each row carries its id, level
count and gold so the alignment can be checked rather than assumed.

Usage:
    python model/scripts/base_native_scores.py \
        --config model/configs/r2_reranker4b.yaml \
        --base-model Qwen/Qwen3-Reranker-4B \
        --data-root /home/pding/projects/open-decisions \
        --json model/experiments/base_native_s123_ep10.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data.pipeline as pipeline_mod  # noqa: E402
import data.sources as sources_mod  # noqa: E402
from data.pipeline import render_jevbench_state  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.evaluation.scorer_blend import sample_choice_items  # noqa: E402
from model.src.reranker_prompt import score_options_native  # noqa: E402


def log_odds(score: float, floor: float = 1e-6) -> float:
    """Turn an independent per-option probability into a logit."""
    return math.log(max(score, floor)) - math.log(max(1.0 - score, floor))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--long-items", type=int, default=150)
    parser.add_argument("--short-max", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.data_root:
        sources_mod.ROOT = args.data_root
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"

    # --- Items, in blend_scorers.py's exact order --------------------------
    rng = random.Random(args.seed)
    items: list[tuple[str, Any]] = []
    for source_name, source_cfg in (cfg.get("sources") or {}).items():
        _, eval_items = load_source_items(source_name, source_cfg or {}, rng)
        source_choice = [item for item in eval_items if item.question.get("type") == "choice"]
        sampled = sample_choice_items(
            source_choice, args.per_source, args.long_items, random.Random(args.seed)
        )
        items.extend(("in-distribution", item) for item in sampled)
    # JevBench is appended last, in tier order, exactly as the blend run does.
    jevbench_dir = (pipeline_mod.DATA_DIR) / "jevbench"
    for tier in ("easy", "hard", "original"):
        path = jevbench_dir / f"{tier}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                if task["question"].get("type") != "choice":
                    continue
                task["state"] = render_jevbench_state(task["state"])
                items.append((f"jevbench-{tier}", task))

    print(f"{len(items)} items in blend order", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # No device_map: accelerate is not installed on the training hosts, and a
    # single-GPU load does not need it.
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    backbone.to(args.device)
    backbone.eval()

    rows: list[dict[str, Any]] = []
    for index, (group, item) in enumerate(items, start=1):
        state = item.state if hasattr(item, "state") else item["state"]
        question = item.question if hasattr(item, "question") else item["question"]
        label = item.label if hasattr(item, "label") else item.get("expected")
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            keys = list(criteria)
            texts = [value or key for key, value in criteria.items()]
        elif isinstance(criteria, list):
            keys = [str(position) for position in range(len(criteria))]
            texts = [str(value) for value in criteria]
        else:
            continue
        if isinstance(label, str):
            if label not in keys:
                continue
            gold = keys.index(label)
        else:
            try:
                gold = int(label)
            except (TypeError, ValueError):
                continue
            if not 0 <= gold < len(keys):
                continue

        scores = score_options_native(
            backbone,
            tokenizer,
            state,
            question,
            list(zip(keys, texts)),
            max_length=args.max_length,
            device=args.device,
        )
        rows.append(
            {
                "group": group,
                "id": item.id if hasattr(item, "id") else item.get("id"),
                "k": len(keys),
                "gold": gold,
                "base": [log_odds(value) for value in scores],
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(items)} scored", flush=True)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"rows": rows}, indent=None) + "\n", encoding="utf-8")
    print(f"Rows saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
