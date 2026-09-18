#!/usr/bin/env python3
"""Probe 5: Cross-model comparison — Jev vs known causal models.

Runs the SAME ordering bias and code-lookup tests through Jev AND through
a known causal model (GPT-4.1-mini via argo proxy) using structured output
prompts that mimic Jev's choice/noul interface.

This is the definitive test: if Jev's recency bias matches a known causal
model's bias, Jev is causal. If significantly less, something non-standard.
"""

import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from httpclient import Client
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "05-cross-model.jsonl")

random.seed(42)

ARGO_BASE = os.environ.get("ARGO_BASE_URL", "http://localhost:44500")
ARGO_MODEL = os.environ.get("ARGO_MODEL", "GPT-4.1-mini")

# --- Shared test cases ---

ORDERING_CASES = [
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

CODE_LOOKUP_CASES = [
    {"n": 1, "codes": [("ZX7", "urgent")]},
    {"n": 3, "codes": [("ZX7", "urgent"), ("QM3", "billing"), ("KP9", "refund")]},
    {"n": 5, "codes": [("ZX7", "urgent"), ("QM3", "billing"), ("KP9", "refund"), ("VN2", "shipping"), ("WT5", "complaint")]},
    {"n": 10, "codes": [("ZX7", "urgent"), ("QM3", "billing"), ("KP9", "refund"), ("VN2", "shipping"), ("WT5", "complaint"), ("HJ8", "technical"), ("BF4", "escalation"), ("RL6", "positive"), ("DS1", "cancellation"), ("YC0", "duplicate")]},
]


class ArgoChoiceClient:
    """Calls a known causal LLM via argo proxy with structured output to mimic Jev's choice interface."""

    def __init__(self):
        self._client = Client(
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

    def choice(self, state: str, instructions: str, options: dict) -> dict:
        option_list = "\n".join(f"- {k}: {v}" for k, v in options.items())
        prob_fields = ", ".join('"' + k + '": <0-1>' for k in options)
        example = '{"choice": "<option_key>", "probabilities": {' + prob_fields + '}}'
        prompt = (
            f"You are evaluating the following state and must choose exactly one option.\n\n"
            f"State: {state}\n\n"
            f"Question: {instructions}\n\n"
            f"Options:\n{option_list}\n\n"
            f"Respond with ONLY a JSON object with this exact format:\n"
            f"{example}\n"
            f"The probabilities must sum to 1.0. Use your best judgment."
        )
        payload = {
            "model": ARGO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        }
        resp = self._client.post(f"{ARGO_BASE}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Argo API error {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)

    def noul(self, state: str, instructions: str) -> float:
        prompt = (
            f"You are evaluating the following state.\n\n"
            f"State: {state}\n\n"
            f"Question: {instructions}\n\n"
            f"Respond with ONLY a JSON object: {{\"noul\": <probability 0-1>}}\n"
            f"The noul value is the probability that the answer is YES."
        )
        payload = {
            "model": ARGO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 50,
        }
        resp = self._client.post(f"{ARGO_BASE}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Argo API error {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)["noul"]

    def close(self):
        self._client.close()


def build_code_state(codes, target_idx, placement):
    target_code, target_meaning = codes[target_idx]
    defs = "\n".join(f"  {c} = {m}" for c, m in codes)
    usage = f"Customer ticket is tagged: {target_code}"
    if placement == "def_first":
        return f"Code definitions:\n{defs}\n\n{usage}"
    else:
        return f"{usage}\n\nCode definitions:\n{defs}"


def main():
    jev = JevClient()
    argo = ArgoChoiceClient()
    records = []
    open(OUTFILE, "w").close()

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    print("=" * 70)
    print(f"PROBE 5: Cross-Model Comparison (Jev vs {ARGO_MODEL})")
    print("=" * 70)

    # --- Part A: Ordering bias comparison ---
    print(f"\n--- Part A: Ordering bias ({len(ORDERING_CASES)} cases × 6 perms × 2 models) ---")

    for ci, case in enumerate(ORDERING_CASES):
        options = case["options"]
        keys = list(options.keys())
        perms = list(itertools.permutations(keys))

        for perm in perms:
            ordered_opts = {k: options[k] for k in perm}

            # Jev
            t0 = time.monotonic()
            jev_resp = jev.choice(case["state"], "q", case["instructions"], ordered_opts)
            jev_time = time.monotonic() - t0
            jev_ans = jev_resp["answers"]["q"]

            # Argo (known causal)
            t0 = time.monotonic()
            try:
                argo_ans = argo.choice(case["state"], case["instructions"], ordered_opts)
                argo_time = time.monotonic() - t0
            except Exception as e:
                argo_ans = {"choice": "ERROR", "probabilities": {k: 0 for k in keys}}
                argo_time = time.monotonic() - t0
                print(f"    Argo error: {e}")

            emit({
                "test": "ordering",
                "case_idx": ci,
                "option_order": list(perm),
                "jev_choice": jev_ans["choice"],
                "jev_probabilities": jev_ans["probabilities"],
                "jev_confidence": jev_ans["confidence"],
                "jev_latency": round(jev_time, 4),
                "argo_choice": argo_ans.get("choice", "ERROR"),
                "argo_probabilities": argo_ans.get("probabilities", {}),
                "argo_latency": round(argo_time, 4),
            })

        print(f"  Case {ci}: done")

    # --- Part B: Code-lookup comparison ---
    print(f"\n--- Part B: Code-lookup def_first vs def_last ({len(CODE_LOOKUP_CASES)} N-values × 2 placements × 2 models) ---")

    for cl_case in CODE_LOOKUP_CASES:
        n = cl_case["n"]
        codes = cl_case["codes"][:n]
        target_idx = 0
        target_meaning = codes[target_idx][1]
        question = f"Is this ticket about {target_meaning}?"

        for placement in ["def_first", "def_last"]:
            state = build_code_state(codes, target_idx, placement)

            # Jev
            t0 = time.monotonic()
            jev_resp = jev.noul(state, "q", question)
            jev_time = time.monotonic() - t0
            jev_noul = jev_resp["answers"]["q"]["noul"]

            # Argo
            t0 = time.monotonic()
            try:
                argo_noul = argo.noul(state, question)
                argo_time = time.monotonic() - t0
            except Exception as e:
                argo_noul = -1
                argo_time = time.monotonic() - t0
                print(f"    Argo error: {e}")

            emit({
                "test": "code_lookup",
                "n_codes": n,
                "placement": placement,
                "jev_noul": jev_noul,
                "jev_latency": round(jev_time, 4),
                "argo_noul": argo_noul,
                "argo_latency": round(argo_time, 4),
            })

        print(f"  N={n}: done")

    jev.close()
    argo.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS: Jev vs Known Causal Model")
    print("=" * 70)

    # A: Position bias comparison
    ordering = [r for r in records if r["test"] == "ordering"]
    print(f"\n--- Ordering bias comparison ({len(ordering)} records) ---")
    for pos in range(3):
        jev_probs = []
        argo_probs = []
        for r in ordering:
            opt = r["option_order"][pos]
            jev_probs.append(r["jev_probabilities"].get(opt, 0))
            if r["argo_probabilities"]:
                argo_probs.append(r["argo_probabilities"].get(opt, 0))
        jev_mean = sum(jev_probs) / len(jev_probs) if jev_probs else 0
        argo_mean = sum(argo_probs) / len(argo_probs) if argo_probs else 0
        expected = 1 / 3
        print(f"  Position {pos}: Jev={jev_mean:.4f} ({jev_mean - expected:+.4f})  {ARGO_MODEL}={argo_mean:.4f} ({argo_mean - expected:+.4f})")

    # B: Code-lookup comparison
    code_lookup = [r for r in records if r["test"] == "code_lookup"]
    print(f"\n--- Code-lookup: def_first vs def_last ---")
    print(f"{'N':>3s} | {'Jev first':>9s} | {'Jev last':>8s} | {'Jev Δ':>6s} | {ARGO_MODEL+' first':>16s} | {ARGO_MODEL+' last':>15s} | {ARGO_MODEL+' Δ':>12s}")
    print("-" * 85)
    for n in sorted(set(r["n_codes"] for r in code_lookup)):
        jf = [r for r in code_lookup if r["n_codes"] == n and r["placement"] == "def_first"]
        jl = [r for r in code_lookup if r["n_codes"] == n and r["placement"] == "def_last"]
        jfn = jf[0]["jev_noul"] if jf else 0
        jln = jl[0]["jev_noul"] if jl else 0
        afn = jf[0]["argo_noul"] if jf else 0
        aln = jl[0]["argo_noul"] if jl else 0
        print(f"{n:3d} | {jfn:9.3f} | {jln:8.3f} | {jln - jfn:+6.3f} | {afn:16.3f} | {aln:15.3f} | {aln - afn:+12.3f}")

    # Agreement
    jev_choices = [r["jev_choice"] for r in ordering]
    argo_choices = [r["argo_choice"] for r in ordering]
    agree = sum(1 for j, a in zip(jev_choices, argo_choices) if j == a)
    print(f"\nChoice agreement: {agree}/{len(ordering)} ({100*agree/len(ordering):.0f}%)")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
