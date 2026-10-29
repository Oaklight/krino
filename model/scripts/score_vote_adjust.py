#!/usr/bin/env python3
"""Can a class-prior correction fix the score head's scale mismatch?

The 4B epoch-10 checkpoint's score head tracks the marginal it was trained on
rather than the marginal of the split it is scored on. On six-level items (STS-B)
the gold split is 53.4% level 0 while the training split is 10.7% and the head
predicts 14.0%; on four-level items it under-predicts level 0 and puts the
surplus one level up. Neither a temperature nor any monotone confidence map
changes an argmax, so calibration cannot repair either. A class prior can, and
this measures whether it does.

For each score item the script takes the argmax twice, once raw and once after
the logit shift from ``model/evaluation/prior_adjustment.py``:

    adjusted_i = logit_i + alpha * (log p_ref_i - log p_hat_i)

where ``p_hat`` is the marginal the head produces on the items being scored and
``p_ref`` is the marginal the correction aims at. ``alpha`` is swept, so the
report shows where QWK peaks instead of assuming a full correction is right.

Two references are measured, because they answer different questions:

* ``train`` -- the label marginal of the training split. Deployable: it is known
  before any eval item is seen.
* ``gold`` -- the label marginal of the eval items themselves. An oracle, and
  therefore a bound on what any prior correction could reach on those items.

The ``p_hat`` term comes from the batch being scored, so this is a transductive
correction. It uses the model's own outputs on those items, never their labels.

Usage:
    python model/scripts/score_vote_adjust.py \
        --config model/configs/r2_reranker4b.yaml \
        --load-heads model/checkpoints/arch_4b_loraAll_32k_s123/epoch_010.pt \
        --data-root /home/pding/projects/open-decisions \
        --lora-layers -1 --readout last --json model/experiments/score_vote_adjust.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data.pipeline as pipeline_mod
import data.sources as sources_mod
from data.sources import load_source_items
from model.evaluation.prior_adjustment import (
    MIN_SCALE_ITEMS,
    adjusted_argmax,
    marginal,
    qwk_by_scale,
)
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info
from model.src.model_factory import build_decision_model


def score_gold_index(item: Any, n_levels: int) -> int:
    """Gold level of a score item, clamped to the level list.

    Score labels are the level itself (a number), not a key into ``criteria``.
    ``_predict_original`` in ``model/scripts/eval_calibration.py`` rounds and
    clamps the same way, and the two must agree or the two reports describe
    different items.
    """
    return max(0, min(n_levels - 1, int(round(float(item.label)))))


def argmax_by_scale(logits_by_scale: dict[int, list[list[float]]]) -> dict[int, list[int]]:
    """Uncorrected argmax per scale, which is also the ``p_hat`` the shift needs."""
    return {
        n_levels: [
            max(range(n_levels), key=lambda index: (row[index], -index)) for row in rows
        ]
        for n_levels, rows in logits_by_scale.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sources", default=None, help="Comma list; default is every config source")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--readout-n", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-targets", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--noul-rank", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--dump-logits", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

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
    alphas = [float(part) for part in args.alphas.split(",") if part.strip()]

    if args.data_root:
        sources_mod.ROOT = args.data_root
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"

    sources_cfg = cfg.get("sources") or {}
    if not sources_cfg:
        raise SystemExit("Error: no sources defined in config")
    wanted = None
    if args.sources:
        wanted = {name.strip() for name in args.sources.split(",") if name.strip()}
        unknown = wanted - set(sources_cfg)
        if unknown:
            raise SystemExit(f"Error: unknown sources: {sorted(unknown)}")

    print("=== Score vote adjustment ===", flush=True)
    print(f"Heads: {args.load_heads}", flush=True)

    # --- Items -------------------------------------------------------------
    # The same rng, the same config order and the same load_source_items call as
    # eval_calibration, so the eval items here are the items that produced the
    # QWK this experiment is trying to move. The train split comes back from the
    # same call and is the marginal a deployment could know.
    rng = random.Random(args.seed)
    score_items: list[Any] = []
    train_gold: dict[int, list[int]] = defaultdict(list)
    for source_name, source_cfg in sources_cfg.items():
        if wanted is not None and source_name not in wanted:
            continue
        train_items, eval_items = load_source_items(source_name, source_cfg or {}, rng)
        for item in train_items:
            if item.question.get("type") != "score":
                continue
            n_levels = len(item.question["criteria"])
            train_gold[n_levels].append(score_gold_index(item, n_levels))
        source_score = [item for item in eval_items if item.question.get("type") == "score"]
        score_items.extend(source_score)
        if source_score:
            print(f"  {source_name:<20} {len(source_score):>5} score eval items", flush=True)

    if not score_items:
        raise SystemExit("Error: no score items in the eval split")

    train_prior = {k: marginal(v, k) for k, v in train_gold.items()}
    print(f"Training prior by level count: {json.dumps(train_prior, sort_keys=True)}", flush=True)

    # --- Forward pass ------------------------------------------------------
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
        readout_n=int(args.readout_n or cfg.get("readout_n") or 4),
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )
    model.load_heads(args.load_heads)
    model.eval()

    logits_by_scale: dict[int, list[list[float]]] = defaultdict(list)
    gold_by_scale: dict[int, list[int]] = defaultdict(list)
    id_by_scale: dict[int, list[str]] = defaultdict(list)
    with torch.no_grad():
        for index, item in enumerate(score_items, start=1):
            criteria = item.question["criteria"]
            n_levels = len(criteria)
            logits = model.forward_score(
                item.state, item.question.get("instructions", ""), criteria
            )[0]
            logits_by_scale[n_levels].append(logits.float().cpu().tolist())
            gold_by_scale[n_levels].append(score_gold_index(item, n_levels))
            id_by_scale[n_levels].append(item.id)
            if index % 200 == 0:
                print(f"  {index}/{len(score_items)} scored", flush=True)

    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"id": item_id, "k": n_levels, "gold": gold, "logits": logits}
            for n_levels, rows in logits_by_scale.items()
            for logits, gold, item_id in zip(
                rows, gold_by_scale[n_levels], id_by_scale[n_levels]
            )
        ]
        args.dump_logits.write_text(json.dumps({"items": payload}) + "\n", encoding="utf-8")
        print(f"Logits saved to {args.dump_logits}", flush=True)

    # --- Priors ------------------------------------------------------------
    gold_prior = {k: marginal(v, k) for k, v in gold_by_scale.items()}
    argmax_scale = argmax_by_scale(logits_by_scale)
    observed_prior = {
        n_levels: marginal(picks, n_levels) for n_levels, picks in argmax_scale.items()
    }

    print("\nMarginals by level count (train / model argmax / gold):", flush=True)
    for n_levels in sorted(gold_by_scale):
        print(
            f"  k={n_levels} n={len(gold_by_scale[n_levels]):5d}\n"
            f"    train {[round(p, 3) for p in train_prior.get(n_levels, [])]}\n"
            f"    model {[round(p, 3) for p in observed_prior[n_levels]]}\n"
            f"    gold  {[round(p, 3) for p in gold_prior[n_levels]]}",
            flush=True,
        )

    # --- Sweep -------------------------------------------------------------
    report: dict[str, Any] = {}
    for reference in ("train", "gold"):
        prior = train_prior if reference == "train" else gold_prior
        curves: dict[str, Any] = {}
        for alpha in alphas:
            predictions: dict[int, list[int]] = {}
            for n_levels, rows in logits_by_scale.items():
                if n_levels not in prior:
                    predictions[n_levels] = argmax_scale[n_levels]
                    continue
                predictions[n_levels] = [
                    adjusted_argmax(row, observed_prior[n_levels], prior[n_levels], alpha)
                    for row in rows
                ]
            curves[str(alpha)] = qwk_by_scale(predictions, gold_by_scale)
            summary = curves[str(alpha)]
            per_scale = " ".join(
                f"k{k}={group['qwk']:.3f}"
                for k, group in summary["by_scale"].items()
                if group["n"] >= MIN_SCALE_ITEMS
            )
            print(
                f"\n[{reference}] alpha={alpha:<5} mixed QWK={summary['mixed_scale_qwk']:.4f} "
                f"acc={summary['mixed_scale_accuracy']:.4f}  {per_scale}",
                flush=True,
            )
        best_alpha = max(curves, key=lambda a: curves[a]["mixed_scale_qwk"])
        report[reference] = {
            "prior": {str(k): v for k, v in prior.items()},
            "curves": curves,
            "best_alpha": float(best_alpha),
            "best_mixed_scale_qwk": curves[best_alpha]["mixed_scale_qwk"],
        }

    payload = {
        "checkpoint": str(args.load_heads),
        "config": str(args.config),
        "seed": args.seed,
        "n_score_items": len(score_items),
        "min_scale_items": MIN_SCALE_ITEMS,
        "observed_argmax_prior": {str(k): v for k, v in observed_prior.items()},
        "gold_prior": {str(k): v for k, v in gold_prior.items()},
        "train_prior": {str(k): v for k, v in train_prior.items()},
        "baseline": qwk_by_scale(argmax_scale, gold_by_scale),
        "references": report,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
