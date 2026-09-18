#!/usr/bin/env python3
"""Probe 5c: Direct logit ordering bias — our own Jev-like scorer.

Loads Qwen3.5-4B, scores options by reading logprobs directly (Route B),
then measures ordering bias on the SAME cases as Jev Probe 4b.
This is the clean comparison: same model family, same test cases,
direct logit readout (no prompted probabilities).

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 probe_05c_direct_logit_ordering.py
"""

import itertools
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "05c-direct-logit-ordering.jsonl")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")

CASES = [
    {
        "state": "I bought these shoes last month and they're starting to fall apart. What can you do?",
        "instructions": "What does the customer want?",
        "options": {"refund": "Money back", "exchange": "Different product", "repair": "Fix current item"},
    },
    {
        "state": "The API sometimes returns a 500 error. It seems to happen about once an hour.",
        "instructions": "How severe is this issue?",
        "options": {"low": "Minor, intermittent", "medium": "Significant but workaround exists", "high": "Blocking"},
    },
    {
        "state": "Your product works great for small files but chokes on anything over 1GB.",
        "instructions": "What is this about?",
        "options": {"performance": "Speed and scaling", "limitation": "Product constraints", "bug": "Broken behavior"},
    },
    {
        "state": "The new dashboard is nice but I really miss the old export feature.",
        "instructions": "What type of message is this?",
        "options": {"feature_request": "Wants new feature", "regression": "Broken feature", "feedback": "General opinion"},
    },
    {
        "state": "I've been a customer for years and I'm honestly disappointed with the recent changes.",
        "instructions": "What is the primary emotion?",
        "options": {"sadness": "Feeling let down", "anger": "Feeling annoyed", "disappointment": "Unmet expectations"},
    },
    {
        "state": "Can you help me understand the pricing for teams? We might be interested in upgrading.",
        "instructions": "Which team handles this?",
        "options": {"sales": "Revenue and pricing", "support": "Customer help", "billing": "Payment and invoices"},
    },
]


def score_options(model, tokenizer, context: str, options: list[str], device: str = "cuda") -> list[float]:
    """Score options by reading log-probabilities of option tokens given context.

    For each option, compute the mean log-probability of its tokens
    conditioned on the context. Returns softmax-normalized probabilities.
    """
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids[0].to(device)

    with torch.inference_mode():
        ctx_out = model(ctx_ids[None], use_cache=True)
        cache = ctx_out.past_key_values
        last_logit = ctx_out.logits[0, -1]

    scores = []
    for opt_text in options:
        opt_ids = tokenizer(opt_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

        with torch.inference_mode():
            # Clone cache for each option (DynamicCache)
            import copy
            opt_cache = copy.deepcopy(cache)
            opt_out = model(opt_ids[None], past_key_values=opt_cache, use_cache=True)

        # Log-probs: last context logit predicts first option token,
        # then each option position predicts the next
        logits = torch.cat([last_logit[None], opt_out.logits[0, :-1]], dim=0)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp[torch.arange(len(opt_ids)), opt_ids]
        scores.append(tok_lp.mean().item())

    # Softmax normalize
    scores_tensor = torch.tensor(scores)
    probs = torch.softmax(scores_tensor, dim=0).tolist()
    return probs


def main():
    print("=" * 70)
    print(f"PROBE 5c: Direct Logit Ordering Bias ({MODEL_NAME})")
    print("=" * 70)

    print(f"\nLoading model...")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    records = []
    open(OUTFILE, "w").close()

    for ci, case in enumerate(CASES):
        keys = list(case["options"].keys())
        perms = list(itertools.permutations(keys))

        # Build context prompt
        for perm in perms:
            option_list = "\n".join(f"- {k}: {case['options'][k]}" for k in perm)
            context = (
                f"State: {case['state']}\n"
                f"Question: {case['instructions']}\n"
                f"Options:\n{option_list}\n"
                f"Answer:"
            )
            # Score each option key as a continuation
            option_texts = [f" {k}" for k in perm]

            t0 = time.monotonic()
            probs = score_options(model, tokenizer, context, option_texts)
            elapsed = time.monotonic() - t0

            prob_dict = {k: round(p, 6) for k, p in zip(perm, probs)}
            choice = max(prob_dict, key=prob_dict.get)

            record = {
                "case_idx": ci,
                "option_order": list(perm),
                "choice": choice,
                "probabilities": prob_dict,
                "elapsed_s": round(elapsed, 4),
                "model": MODEL_NAME,
            }
            records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

        print(f"  Case {ci}: done ({len(perms)} perms)")

    # --- Position Bias Analysis ---
    print(f"\n{'=' * 70}")
    print("POSITION BIAS ANALYSIS")
    print("=" * 70)

    for pos in range(3):
        probs_at_pos = []
        for r in records:
            opt = r["option_order"][pos]
            probs_at_pos.append(r["probabilities"][opt])
        mean = sum(probs_at_pos) / len(probs_at_pos)
        print(f"  Position {pos}: mean prob = {mean:.4f} (delta = {mean - 1/3:+.4f})")

    # Compare
    print("\n--- Comparison ---")
    print("  Jev (Probe 5):     pos0=-0.016  pos1=+0.014  pos2=+0.001")
    print("  GPT-4.1-mini:      pos0=-0.036  pos1=+0.040  pos2=-0.004")
    print(f"  {MODEL_NAME}:  (see above)")

    # Flip rate
    flip_count = 0
    for ci in range(len(CASES)):
        cr = [r for r in records if r["case_idx"] == ci]
        if len(set(r["choice"] for r in cr)) > 1:
            flip_count += 1
    print(f"\nChoice flip rate: {flip_count}/{len(CASES)} ({100*flip_count/len(CASES):.0f}%)")
    print(f"Total: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
