#!/usr/bin/env python3
"""Does blending the native reranker score with the trained head beat both?

``research/13`` found the two scorers trade places at about twenty options: the
backbone's own per-option P(yes) wins on short lists and loses badly on long
ones, and the trained choice head does the opposite. A hard switch at five
options captures most of that, but a switch throws away whichever half of the
signal the losing scorer had on each item.

This measures the cheapest version of the fix the note proposes -- a weighted
blend of the two score vectors -- before anything is trained on it. It fits one
free parameter on the in-distribution eval and then scores JevBench with it,
which the leakage audit (``scripts/check_jevbench_leakage.py``) shows is
disjoint from training.

The blend is on a common scale. Each scorer's logits are divided by one number,
the median of its own per-item margin on the fitting set, so the blend weight
means the same thing on a two-option item as on an eighty-option one and a
single weight can be read across buckets:

    blended_i = head_i / spread_head + lam * native_i / spread_native

``lam = 0`` is the head alone, ``lam -> infinity`` the native scorer alone.

Scoring is the expensive half: the native scorer runs one forward pass per
option, so ``--dump-records`` writes every per-item score vector and
``--from-records`` re-runs the sweep over them without touching a GPU.

Usage:
    python model/scripts/blend_scorers.py \
        --config model/configs/r2_reranker4b.yaml \
        --load-heads model/checkpoints/arch_4b_loraAll_32k_s123/epoch_010.pt \
        --data-root /home/pding/projects/open-decisions \
        --lora-layers -1 --readout last \
        --dump-records model/experiments/blend_records_s123_ep10.json \
        --json model/experiments/blend_scorers.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.scorer_blend import (  # noqa: E402
    blended_argmax,
    bucket_of,
    margin,
    sample_choice_items,
    spread,
)


def log_odds(score: float, floor: float = 1e-6) -> float:
    """Turn an independent per-option probability into a logit."""
    return math.log(max(score, floor)) - math.log(max(1.0 - score, floor))


def accuracy(pairs: Sequence[tuple[bool, float]]) -> dict[str, Any]:
    """Accuracy and chance floor over (correct, chance) pairs."""
    if not pairs:
        return {"n": 0, "accuracy": 0.0, "chance": 0.0}
    return {
        "n": len(pairs),
        "accuracy": sum(1 for correct, _ in pairs if correct) / len(pairs),
        "chance": sum(chance for _, chance in pairs) / len(pairs),
    }


def option_texts(question: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    """Keys and texts of a choice question, in presentation order."""
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        keys = list(criteria)
        return keys, [value or key for key, value in criteria.items()]
    if isinstance(criteria, list):
        return [str(position) for position in range(len(criteria))], [str(v) for v in criteria]
    return None


def score_choice_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Score every sampled choice item with both the head and the native readout."""
    import torch
    import yaml

    import data.pipeline as pipeline_mod
    import data.sources as sources_mod
    from data.pipeline import render_jevbench_state
    from data.sources import load_source_items
    from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info
    from model.src.model_factory import build_decision_model
    from model.src.reranker_prompt import score_options_native

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_name = args.model or cfg.get("model")
    max_length = int(args.max_length or cfg.get("max_length") or 32768)
    readout = args.readout or cfg.get("readout", "last")
    lora_layers = args.lora_layers if args.lora_layers is not None else cfg.get("lora_layers")
    lora_rank = int(args.lora_rank or cfg.get("lora_rank") or 16)
    lora_alpha = int(args.lora_alpha or cfg.get("lora_alpha") or 32)
    lora_dropout = float(
        args.lora_dropout if args.lora_dropout is not None else cfg.get("lora_dropout", 0.0)
    )
    lora_targets = tuple(
        part.strip()
        for part in str(args.lora_targets or cfg.get("lora_targets") or "q_proj,v_proj").split(",")
        if part.strip()
    )
    rank = int(args.rank or cfg.get("rank") or 128)
    noul_rank = int(args.noul_rank or cfg.get("noul_rank") or rank)
    mlp_layers = int(args.mlp_layers if args.mlp_layers is not None else cfg.get("mlp_layers", 0))

    if args.data_root:
        sources_mod.ROOT = args.data_root
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"
    data_dir = pipeline_mod.DATA_DIR

    # --- Items -------------------------------------------------------------
    # The native scorer costs one forward pass per option, so the long bucket --
    # banking77 at 80 options, the only source above 19 -- is roughly twenty
    # times more expensive per item than a short-list one. Sampling per source
    # keeps the sweep inside a fifteen-minute run while leaving every bucket
    # represented.
    rng = random.Random(args.seed)
    choice_items: list[tuple[str, Any]] = []
    for source_name, source_cfg in ({} if args.jevbench_only else (cfg.get("sources") or {})).items():
        _, eval_items = load_source_items(source_name, source_cfg or {}, rng)
        source_choice = [item for item in eval_items if item.question.get("type") == "choice"]
        sampled = (
            source_choice
            if args.all_items
            else sample_choice_items(
                source_choice, args.per_source, args.long_items, random.Random(args.seed)
            )
        )
        choice_items.extend(("in-distribution", item) for item in sampled)
        if source_choice:
            print(
                f"  {source_name:<20} {len(sampled):>4}/{len(source_choice):<4} choice items",
                flush=True,
            )
    print(f"in-distribution choice items: {len(choice_items)}", flush=True)

    jevbench_dir = args.jevbench_dir or (data_dir / "jevbench")
    for tier in ("easy", "hard", "original"):
        path = jevbench_dir / f"{tier}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} missing, skipping tier", file=sys.stderr, flush=True)
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                if task["question"].get("type") != "choice":
                    continue
                # Same rendering the loader applies, so a structured hard-tier
                # state reaches the model as text rather than as Python repr.
                task["state"] = render_jevbench_state(task["state"])
                choice_items.append((f"jevbench-{tier}", task))
    print(f"total choice items: {len(choice_items)}", flush=True)

    # --- Model -------------------------------------------------------------
    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=max_length)
    print_gpu_info(gpu_config)

    model = build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=args.device or cfg.get("device"),
        use_encoder=bool(cfg.get("encoder", False)),
        rank=rank,
        mlp_layers=mlp_layers,
        noul_rank=noul_rank,
        max_length=max_length,
        shared_attention=bool(cfg.get("shared_attention", True)),
        readout=readout,
        readout_n=int(cfg.get("readout_n") or 4),
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )
    model.load_heads(args.load_heads)
    model.eval()
    native_max_length = min(int(getattr(model, "max_length", 8192) or 8192), 8192)
    device = str(next(model.backbone.parameters()).device)

    # --- Score both ways ---------------------------------------------------
    records: list[dict[str, Any]] = []
    for index, (group, item) in enumerate(choice_items, start=1):
        state = item.state if hasattr(item, "state") else item["state"]
        question = item.question if hasattr(item, "question") else item["question"]
        label = item.label if hasattr(item, "label") else item.get("expected")
        resolved = option_texts(question)
        if resolved is None:
            continue
        keys, texts = resolved
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

        with torch.no_grad():
            head = model.forward_choice(state, question.get("instructions", ""), texts)[0]
        # The rule keeps the head for long lists, so scoring them natively costs
        # one forward pass per option for a score nothing reads.
        if args.skip_native_long and len(keys) >= args.skip_native_long:
            native = None
        else:
            native = [
                log_odds(value)
                for value in score_options_native(
                    model.backbone,
                    model.tokenizer,
                    state,
                    question,
                    list(zip(keys, texts)),
                    max_length=native_max_length,
                    device=device,
                )
            ]
        records.append(
            {
                "group": group,
                "k": len(keys),
                "gold": gold,
                "head": head.float().cpu().tolist(),
                "native": native,
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(choice_items)} scored", flush=True)
    return records


def sweep(records: Sequence[dict[str, Any]], lams: Sequence[float]) -> dict[str, Any]:
    """Blend weight sweep, with the scale fitted on in-distribution items only."""
    scored = [record for record in records if record.get("native") is not None]
    skipped = len(records) - len(scored)
    if skipped:
        print(f"  {skipped} items have no native score and are left out of the sweep", flush=True)
    records = scored
    in_dist = [record for record in records if record["group"] == "in-distribution"]
    if not in_dist:
        # A JevBench-only run has no in-distribution items; the endpoints of the
        # sweep are still the two scorers, which is what such a run reads.
        in_dist = [record for record in records if record.get("native")]
    head_scale = spread([margin(record["head"]) for record in in_dist])
    native_scale = spread([margin(record["native"]) for record in in_dist])
    print(f"scales (in-distribution medians): head {head_scale:.4f}, native {native_scale:.4f}")

    results: dict[str, Any] = {}
    for lam in lams:
        per_group: dict[str, Any] = {}
        for group in sorted({record["group"] for record in records}):
            group_records = [record for record in records if record["group"] == group]
            by_bucket: dict[str, list[tuple[bool, float]]] = defaultdict(list)
            for record in group_records:
                guess = blended_argmax(
                    record["head"], record["native"], head_scale, native_scale, lam
                )
                chance = 1.0 / record["k"]
                by_bucket[bucket_of(record["k"])].append((guess == record["gold"], chance))
                by_bucket["all"].append((guess == record["gold"], chance))
            per_group[group] = {
                bucket: accuracy(pairs) for bucket, pairs in sorted(by_bucket.items())
            }
        results[str(lam)] = per_group
        row = " ".join(
            f"{group}:{per_group[group]['all']['accuracy']:.4f}" for group in sorted(per_group)
        )
        print(f"lam={lam:<6} {row}", flush=True)

    return {"head_scale": head_scale, "native_scale": native_scale, "results": results}


def rule_sweep(records: Sequence[dict[str, Any]], lams: Sequence[float]) -> dict[str, Any]:
    """Accuracy of the deployable rule: head on long lists, blend on short ones.

    This is the rule the note recommends, evaluated directly instead of read off
    the blend sweep: an item with at least ``LONG_MIN`` options keeps the head's
    argmax, so it needs no native score at all, and everything below that is
    blended. The scale is fitted on the short items, which are the only ones the
    blend touches.

    Args:
        records: Scored items; ``native`` may be ``None`` where it was skipped.
        lams: Blend weights to evaluate.

    Returns:
        ``{"head_scale", "native_scale", "rules"}`` where each rule holds
        per-bucket and overall accuracy.
    """
    from model.evaluation.scorer_blend import LONG_MIN

    short = [record for record in records if record["k"] < LONG_MIN and record.get("native")]
    if not short:
        raise SystemExit("Error: no short-list records with a native score to fit the scale on")
    head_scale = spread([margin(record["head"]) for record in short])
    native_scale = spread([margin(record["native"]) for record in short])
    print(
        f"rule scales (short-list medians): head {head_scale:.4f}, native {native_scale:.4f}",
        flush=True,
    )

    rules: dict[str, Any] = {}
    for lam in lams:
        per_group: dict[str, Any] = {}
        for group in sorted({record["group"] for record in records}):
            by_bucket: dict[str, list[tuple[bool, float]]] = defaultdict(list)
            for record in records:
                if record["group"] != group:
                    continue
                if record["k"] >= LONG_MIN or not record.get("native"):
                    guess = max(
                        range(record["k"]), key=lambda index: (record["head"][index], -index)
                    )
                else:
                    guess = blended_argmax(
                        record["head"], record["native"], head_scale, native_scale, lam
                    )
                chance = 1.0 / record["k"]
                by_bucket[bucket_of(record["k"])].append((guess == record["gold"], chance))
                by_bucket["all"].append((guess == record["gold"], chance))
            per_group[group] = {
                bucket: accuracy(pairs) for bucket, pairs in sorted(by_bucket.items())
            }
        rules[str(lam)] = per_group
        parts = " ".join(
            f"{group}:{per_group[group]['all']['accuracy']:.4f}({per_group[group]['all']['n']})"
            for group in sorted(per_group)
        )
        print(f"rule lam={lam:<6} {parts}", flush=True)

    return {"head_scale": head_scale, "native_scale": native_scale, "rules": rules}


def checkpoint_seed(path: Path | None) -> int | None:
    """The training seed a checkpoint path names, if it names one.

    ``--seed`` is the sampling seed -- which items a run scores -- and it has a
    default, so a report that calls it ``seed`` reads as if it were the training
    seed even when the run never passed it. The training seed is in the
    checkpoint's name (``arch_4b_32k_split138_s42``); this reads it there, and
    returns ``None`` rather than guessing when the name does not carry one.
    """
    if path is None:
        return None
    match = re.search(r"_s(\d+)(?:/|$)", str(path))
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed of the checkpoint, for names that do not carry one; "
        "recorded in the report so it cannot be confused with --seed",
    )
    parser.add_argument("--lams", default="0,0.125,0.25,0.5,1,2,4,8")
    parser.add_argument(
        "--per-source",
        type=int,
        default=60,
        help="Short-list items sampled per source; the native scorer costs a forward pass per option",
    )
    parser.add_argument(
        "--long-items",
        type=int,
        default=150,
        help="Items with at least 20 options sampled per source, which is only banking77 here",
    )
    parser.add_argument(
        "--all-items", action="store_true", help="Score every choice item instead of a sample"
    )
    parser.add_argument(
        "--skip-native-long",
        type=int,
        default=0,
        help="Do not score items with at least this many options natively (0 = score all)",
    )
    parser.add_argument(
        "--jevbench-only",
        action="store_true",
        help="Score only JevBench; the scale is then fitted on its own items, so read lam=0 and lam=1e9",
    )
    parser.add_argument("--jevbench-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-targets", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--noul-rank", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--dump-records", type=Path, default=None)
    parser.add_argument(
        "--from-records",
        type=Path,
        default=None,
        help="Re-run the sweep over a previous --dump-records file, without a GPU",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    lams = [float(part) for part in args.lams.split(",") if part.strip()]

    if args.from_records:
        records = json.loads(args.from_records.read_text(encoding="utf-8"))["records"]
        print(f"loaded {len(records)} records from {args.from_records}", flush=True)
    else:
        records = score_choice_items(args)
        if args.dump_records:
            args.dump_records.parent.mkdir(parents=True, exist_ok=True)
            args.dump_records.write_text(
                json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"Records saved to {args.dump_records}", flush=True)

    rule = rule_sweep(records, lams)
    analysis = {"rule": rule, **sweep(records, lams)}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.load_heads),
            "checkpoint_seed": args.training_seed
            if args.training_seed is not None
            else checkpoint_seed(args.load_heads),
            "config": str(args.config),
            "sample_seed": args.seed,
            "n_items": len(records),
            "lams": lams,
            **analysis,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
