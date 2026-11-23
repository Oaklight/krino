#!/usr/bin/env python3
"""Probe 5b: Qwen ordering bias via vLLM — direct comparison to Jev.

Runs the EXACT same ordering bias cases from Probe 4b through a Qwen model
served via vLLM's OpenAI-compatible API, using system-one-adapter-style
structured output prompts.

Usage:
  VLLM_BASE=http://localhost:8000 python3 probe_05b_qwen_ordering.py
"""

import itertools
import json
import os
import random
import sys
import time

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_repo_root, "_vendor"))
from httpclient import Client

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "05b-qwen-ordering.jsonl")

random.seed(42)

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
MODEL_NAME = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-4B")

# Same cases as Probe 4b
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


def call_vllm_choice(client: Client, state: str, instructions: str, options: dict) -> dict:
    option_list = "\n".join(f"- {k}: {v}" for k, v in options.items())
    keys = list(options.keys())
    prob_fields = ", ".join('"' + k + '": <probability 0 to 1>' for k in keys)
    example = '{"choice": "<key>", "probabilities": {' + prob_fields + '}}'

    prompt = (
        f"Evaluate this state and choose exactly one option. "
        f"Return ONLY valid JSON.\n\n"
        f"State: {state}\n\n"
        f"Question: {instructions}\n\n"
        f"Options:\n{option_list}\n\n"
        f"Respond with exactly this JSON format:\n{example}\n"
        f"Probabilities must sum to 1.0."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }

    resp = client.post(f"{VLLM_BASE}/v1/chat/completions", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"vLLM error {resp.status_code}: {resp.text[:200]}")

    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Handle cases where model outputs extra text
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]
    return json.loads(content)


def main():
    client = Client(timeout=120)
    records = []
    open(OUTFILE, "w").close()

    # Check model is available
    try:
        resp = client.get(f"{VLLM_BASE}/v1/models")
        models = resp.json()
        print(f"Available models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        print(f"Cannot connect to vLLM at {VLLM_BASE}: {e}")
        sys.exit(1)

    print("=" * 70)
    print(f"PROBE 5b: Qwen Ordering Bias ({MODEL_NAME})")
    print(f"Running same cases as Probe 4b for direct comparison")
    print("=" * 70)

    for ci, case in enumerate(CASES):
        keys = list(case["options"].keys())
        perms = list(itertools.permutations(keys))

        for perm in perms:
            ordered = {k: case["options"][k] for k in perm}
            t0 = time.monotonic()
            try:
                ans = call_vllm_choice(client, case["state"], case["instructions"], ordered)
                elapsed = time.monotonic() - t0
                record = {
                    "case_idx": ci,
                    "option_order": list(perm),
                    "choice": ans.get("choice", "ERROR"),
                    "probabilities": ans.get("probabilities", {}),
                    "elapsed_s": round(elapsed, 4),
                    "model": MODEL_NAME,
                }
            except Exception as e:
                elapsed = time.monotonic() - t0
                record = {
                    "case_idx": ci,
                    "option_order": list(perm),
                    "choice": "ERROR",
                    "probabilities": {},
                    "elapsed_s": round(elapsed, 4),
                    "model": MODEL_NAME,
                    "error": str(e)[:200],
                }

            records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

        print(f"  Case {ci}: done ({len(perms)} perms)")

    client.close()

    # --- Position Bias Analysis ---
    print(f"\n{'=' * 70}")
    print("POSITION BIAS ANALYSIS")
    print("=" * 70)

    valid = [r for r in records if r["choice"] != "ERROR" and r["probabilities"]]
    print(f"Valid records: {len(valid)}/{len(records)}")

    for pos in range(3):
        probs = []
        for r in valid:
            opt = r["option_order"][pos]
            p = r["probabilities"].get(opt, 0)
            probs.append(p)
        if probs:
            mean = sum(probs) / len(probs)
            print(f"  Position {pos}: mean prob = {mean:.4f} (delta from 0.333 = {mean - 1/3:+.4f})")

    # Compare with Jev's Probe 4 results (3-option cases from probe_05)
    print("\n--- Comparison (3-option cases) ---")
    print("  Jev (Probe 5):     pos0=-0.016  pos1=+0.014  pos2=+0.001")
    print(f"  {MODEL_NAME}:  (see above)")

    flip_count = 0
    for ci in range(len(CASES)):
        cr = [r for r in valid if r["case_idx"] == ci]
        if len(set(r["choice"] for r in cr)) > 1:
            flip_count += 1
    print(f"\nChoice flip rate: {flip_count}/{len(CASES)} ({100*flip_count/len(CASES):.0f}%)")
    print(f"Total: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
