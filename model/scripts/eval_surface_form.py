#!/usr/bin/env python3
"""Surface-form sensitivity test for trained decision models.

Reproduces Probe 3 methodology: semantically equivalent option descriptions
with different length, grammaticality, and commonness under Latin-square
balancing. Measures whether the model assigns probability based on meaning
or surface form.

Key metric: R² of description length vs assigned probability.
  R² > 0.3 → LM-logit leakage (surface form matters)
  R² < 0.1 → learned semantic head (meaning matters, not wording)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.backbone import load_causal_lm, load_encoder
from model.src.decision_model import DecisionModel
from model.src.logit_scorer import LogitScorer

# (name, text, length, grammaticality 0/1, commonness 0/1)
# Factorial: grammaticality and commonness are partially crossed to avoid confounding.
# grammatical+common, grammatical+rare, ungrammatical+common, ungrammatical+rare
VARIANTS = [
    ("short_gram_common", "Approve the request.", 20, 1.0, 1.0),
    ("long_gram_rare", "Approve the request pursuant to the heretofore stipulated adjudication.", 72, 1.0, 0.0),
    ("short_ungram_common", "Request approve do.", 19, 0.0, 1.0),
    ("long_ungram_rare", "The request is what shall, following review completion, be approved by action.", 78, 0.0, 0.0),
]

LABEL_SETS = [
    ["opt_a", "opt_b", "opt_c", "opt_d"],
    ["17", "42", "83", "96"],
]

N_ITEMS = 100


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return None
    return cov / (sx * sy)


def run_test(scorer, n_items=N_ITEMS, seed=42):
    lengths, probabilities = [], []
    grammaticalities, commonnesses = [], []
    variant_probs = {v[0]: [] for v in VARIANTS}

    for item_idx in range(n_items):
        for labels in LABEL_SETS:
            shift = item_idx % len(VARIANTS)
            assignment = [(labels[i], VARIANTS[(i + shift) % len(VARIANTS)]) for i in range(len(labels))]
            criteria = {label: variant[1] for label, variant in assignment}
            variant_by_label = {label: variant[0] for label, variant in assignment}
            variant_info_by_label = {label: variant for label, variant in assignment}

            state = f"Case {item_idx}: all listed actions have exactly the same consequence and utility."
            question = {
                "type": "choice",
                "instructions": "Choose among semantically equivalent actions; no action is preferred.",
                "criteria": criteria,
            }

            answer = scorer.predict(state, question)
            probs = answer.get("probabilities", {})

            for label, prob in probs.items():
                variant_name = variant_by_label.get(label)
                if variant_name:
                    variant_info = next(v for v in VARIANTS if v[0] == variant_name)
                    lengths.append(variant_info[2])
                    grammaticalities.append(variant_info[3])
                    commonnesses.append(variant_info[4])
                    probabilities.append(prob)
                    variant_probs[variant_name].append(prob)

    r_length = pearson(lengths, probabilities)
    r_gram = pearson(grammaticalities, probabilities)
    r_common = pearson(commonnesses, probabilities)

    variant_summary = {
        name: {"mean": sum(ps) / len(ps), "n": len(ps)}
        for name, ps in variant_probs.items() if ps
    }

    return {
        "n_items": n_items,
        "n_observations": len(probabilities),
        "length_r_squared": r_length ** 2 if r_length is not None else None,
        "grammaticality_r_squared": r_gram ** 2 if r_gram is not None else None,
        "commonness_r_squared": r_common ** 2 if r_common is not None else None,
        "correlations": {
            "length": r_length,
            "grammaticality": r_gram,
            "commonness": r_common,
        },
        "variant_summary": variant_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Surface-form sensitivity test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoder", action="store_true")
    parser.add_argument("--heads", type=Path, default=None, help="Trained heads checkpoint (omit for logit readout)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--strategy", default="description", choices=["description", "label"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    if args.heads:
        if args.encoder:
            backbone, tokenizer = load_encoder(args.model, device=args.device, freeze=True)
        else:
            backbone, tokenizer = load_causal_lm(args.model, device=args.device, freeze=True)
        model = DecisionModel(backbone=backbone, tokenizer=tokenizer, rank=args.rank)
        model.load_heads(args.heads)
        model.eval()
        scorer = model
        mode = "trained_heads"
    else:
        backbone, tokenizer = load_causal_lm(args.model, device=args.device, freeze=True)
        scorer = LogitScorer(backbone, tokenizer, strategy=args.strategy)
        mode = f"logit_readout_{args.strategy}"

    print(f"Mode: {mode}")
    print(f"Running surface-form test ({args.n_items} items)...")

    t0 = time.monotonic()
    results = run_test(scorer, n_items=args.n_items)
    elapsed = time.monotonic() - t0

    results["mode"] = mode
    results["model"] = args.model
    results["elapsed_s"] = round(elapsed, 1)

    print(f"\n{'='*50}")
    print(f"Surface-form sensitivity:")
    for dim in ("length", "grammaticality", "commonness"):
        r2 = results.get(f"{dim}_r_squared")
        print(f"  {dim:20s} R² = {r2:.4f}" if r2 is not None else f"  {dim:20s} R² = N/A")
    print(f"  (R² > 0.3 → surface form dominates; R² < 0.1 → head ignores it)")
    print(f"\nVariant means:")
    for name, stats in sorted(results["variant_summary"].items()):
        print(f"  {name:15s}: {stats['mean']:.4f} (n={stats['n']})")
    print(f"\nElapsed: {elapsed:.1f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
