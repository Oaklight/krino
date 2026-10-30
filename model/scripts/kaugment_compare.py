#!/usr/bin/env python3
"""Compare two checkpoints' option-count curves on a fixed item set.

Goal item 3 asks for accuracy at >20 options to sit within 5 pp of accuracy at
<=5 options. The k-augment A/B exists to move that gap, and this is the instrument
that reads it: the same items, the same K schedule, the same distractor draws for
both arms, so the only difference between the two columns is the checkpoint.

Running the sweep twice and subtracting by hand would be a mistake the numbers do
not announce — the probes are built from a per-item seed, so two runs that used
different `--seed` values score different option sets and the columns stop being
paired. This script builds the probe list once and scores both checkpoints on it,
then reports the difference per K with an exact McNemar test over the paired
outcomes.

Usage:
    python model/scripts/kaugment_compare.py \
        --config model/configs/r2_reranker0.6b.yaml \
        --control model/checkpoints/arch_06b_kaug50ctl_lr3e4_split138_s123/epoch_005.pt \
        --augmented model/checkpoints/arch_06b_kaug50aug_lr3e4_split138_s123/epoch_005.pt \
        --source banking77 --items 300 --ks 4,8,16,32,64,80 \
        --json model/experiments/kaugment_compare_s123_ep5.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

import data.pipeline as pipeline_mod  # noqa: E402
import data.sources as sources_mod  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info  # noqa: E402
from model.src.model_factory import build_decision_model  # noqa: E402
from model.scripts.option_count_sweep import _at_k, _option_pool  # noqa: E402


def _exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p for ``b`` and ``c`` discordant pairs.

    Duplicated from ``model/evaluation/paired_power.py`` (PR #202) so this script
    does not depend on an unmerged module; replace it with that import once #202
    lands. Under the null both kinds of discordant pair are equally likely, so
    ``b ~ Binomial(b + c, 0.5)``. Assembled in log space because
    ``comb(n, k) * 0.5 ** n`` underflows for the counts a 300-item sweep produces.
    """
    from math import exp, lgamma, log

    n = b + c
    if n == 0 or b == c:
        return 1.0
    tail = sum(
        exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + n * log(0.5))
        for k in range(min(b, c) + 1)
    )
    return min(1.0, 2.0 * tail)


def _build_model(cfg: dict, args: argparse.Namespace) -> Any:
    model_name = args.model or cfg["model"]
    max_length = int(args.max_length or cfg.get("max_length") or 32768)
    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=max_length)
    return build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=args.device or cfg.get("device"),
        use_encoder=bool(cfg.get("encoder", False)),
        rank=int(args.rank or cfg.get("rank") or 128),
        mlp_layers=int(cfg.get("mlp_layers", 0)),
        noul_rank=args.noul_rank if args.noul_rank is not None else cfg.get("noul_rank"),
        max_length=max_length,
        shared_attention=bool(cfg.get("shared_attention", True)),
        readout=args.readout or cfg.get("readout", "last"),
        readout_n=int(args.readout_n or cfg.get("readout_n") or 4),
        lora_layers=args.lora_layers if args.lora_layers is not None else cfg.get("lora_layers"),
        lora_rank=int(cfg.get("lora_rank") or 16),
        lora_alpha=int(cfg.get("lora_alpha") or 32),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        lora_targets=tuple(
            part.strip()
            for part in str(cfg.get("lora_targets") or "q_proj,v_proj").split(",")
            if part.strip()
        ),
        choice_mode=args.choice_mode or str(cfg.get("choice_mode") or "tower"),
    )


def _score_all(model: Any, probes: list[tuple[Any, list[str], list[str]]]) -> list[bool]:
    """Per-probe correctness, in the order the probes were built."""
    outcomes = []
    for item, texts, keys in probes:
        logits = model.forward_choice(item.state, item.question.get("instructions", ""), texts)[0]
        best = int(logits.float().argmax())
        outcomes.append(keys[best] == item.label)
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--source", default="banking77")
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--ks", default="4,8,16,32,64,80")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--readout-n", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--noul-rank", type=int, default=None)
    parser.add_argument("--choice-mode", choices=["tower", "cross_encoder"], default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    if args.data_root:
        sources_mod.ROOT = args.data_root
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"
    data_dir = pipeline_mod.DATA_DIR

    source_cfg = (cfg.get("sources") or {}).get(args.source) or {"path": f"data/benchmarks/{args.source}.jsonl"}
    _, eval_items = load_source_items(args.source, source_cfg, random.Random(args.seed))
    choice_items = [item for item in eval_items if item.question.get("type") == "choice"]
    rng = random.Random(f"{args.seed}:sweep")
    rng.shuffle(choice_items)
    choice_items = choice_items[: args.items]

    pool_texts = _option_pool(data_dir / f"{args.source}.jsonl")
    pool = {args.source: pool_texts}
    ks = [int(part) for part in args.ks.split(",") if part.strip()]

    # Built once, scored by both checkpoints: a probe list rebuilt per model would
    # pair the columns on nothing.
    probes_by_k: dict[int, list[tuple[Any, list[str], list[str]]]] = {}
    for k in ks:
        probes = []
        for item in choice_items:
            probe = _at_k(item, k, pool, random.Random(f"{args.seed}:{k}:{item.id}"))
            criteria = probe.question["criteria"]
            strings = list(criteria.keys())
            texts = [value or key for key, value in criteria.items()]
            probes.append((probe, texts, strings))
        probes_by_k[k] = probes
    print(f"  {len(choice_items)} {args.source} choice items, K in {ks}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for arm, heads in (("control", args.control), ("augmented", args.augmented)):
        print(f"\n=== {arm}: {heads}", flush=True)
        model = _build_model(cfg, args)
        model.load_heads(heads)
        model.eval()
        by_k: dict[str, Any] = {}
        with torch.no_grad():
            for k in ks:
                outcomes = _score_all(model, probes_by_k[k])
                correct = sum(outcomes)
                n = len(outcomes)
                by_k[str(k)] = {
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n if n else 0.0,
                    "chance": 1.0 / k,
                    "outcomes": outcomes,
                }
                print(f"  K={k:<4d} acc={by_k[str(k)]['accuracy']:.4f} ({correct}/{n})", flush=True)
        results[arm] = by_k
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Paired comparison (augmented minus control)", flush=True)
    agreement: dict[str, Any] = {}
    for k in ks:
        ctl = results["control"][str(k)]["outcomes"]
        aug = results["augmented"][str(k)]["outcomes"]
        aug_only = sum(1 for a, b in zip(aug, ctl) if a and not b)
        ctl_only = sum(1 for a, b in zip(aug, ctl) if b and not a)
        delta = results["augmented"][str(k)]["accuracy"] - results["control"][str(k)]["accuracy"]
        p_value = _exact_mcnemar(aug_only, ctl_only)
        agreement[str(k)] = {
            "augmented_accuracy": results["augmented"][str(k)]["accuracy"],
            "control_accuracy": results["control"][str(k)]["accuracy"],
            "delta": delta,
            "augmented_only_right": aug_only,
            "control_only_right": ctl_only,
            "mcnemar_p": p_value,
        }
        print(
            f"  K={k:<4d} aug={results['augmented'][str(k)]['accuracy']:.4f} "
            f"ctl={results['control'][str(k)]['accuracy']:.4f} "
            f"delta={delta:+.4f} ({aug_only}+/{ctl_only}-) p={p_value:.4f}",
            flush=True,
        )

    print("\n=== Clause 3's count half: accuracy at >20 options minus at <=5", flush=True)
    gap: dict[str, Any] = {}
    small = [k for k in ks if k <= 5]
    large = [k for k in ks if k > 20]
    for arm in ("control", "augmented"):
        if not small or not large:
            continue
        low = [results[arm][str(k)]["accuracy"] for k in small]
        high = [results[arm][str(k)]["accuracy"] for k in large]
        small_accuracy = sum(low) / len(low)
        large_accuracy = sum(high) / len(high)
        gap[arm] = {"<=5": small_accuracy, ">20": large_accuracy, "gap": large_accuracy - small_accuracy}
        print(
            f"  {arm:<10s} <=5 {small_accuracy:.4f}  >20 {large_accuracy:.4f}  "
            f"gap {gap[arm]['gap']:+.4f}  ({'passes' if gap[arm]['gap'] >= -0.05 else 'fails'} the 5 pp bar)",
            flush=True,
        )
    if "control" in gap and "augmented" in gap:
        print(f"  gap closed by {gap['augmented']['gap'] - gap['control']['gap']:+.4f}", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "control": str(args.control),
            "augmented": str(args.augmented),
            "source": args.source,
            "items": len(choice_items),
            "ks": ks,
            "seed": args.seed,
            "choice_mode": args.choice_mode or str(cfg.get("choice_mode") or "tower"),
            "per_arm": {
                arm: {k: {key: v for key, v in row.items() if key != "outcomes"} for k, row in by_k.items()}
                for arm, by_k in results.items()
            },
            "paired": agreement,
            "option_count_gap": gap,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
