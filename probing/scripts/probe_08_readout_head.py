#!/usr/bin/env python3
"""Probe 8: Readout head design — per-type behavior differences.

Tests whether noul, choice, and score use the same or different readout heads.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "08-readout-head.jsonl")

CROSS_TYPE_CASES = [
    {
        "state": "The customer is furious and demanding a refund immediately.",
        "question": "Is the customer angry?",
        "expected_noul_high": True,
    },
    {
        "state": "Everything arrived on time and in perfect condition. Thank you!",
        "question": "Is the customer satisfied?",
        "expected_noul_high": True,
    },
    {
        "state": "I have a general question about your pricing.",
        "question": "Is the customer angry?",
        "expected_noul_high": False,
    },
    {
        "state": "Our entire production system is down. We are losing revenue.",
        "question": "Is this a critical issue?",
        "expected_noul_high": True,
    },
    {
        "state": "I noticed a small typo on your about page.",
        "question": "Is this a critical issue?",
        "expected_noul_high": False,
    },
    {
        "state": "I want to cancel my subscription and get a full refund.",
        "question": "Is this a cancellation request?",
        "expected_noul_high": True,
    },
    {
        "state": "Can you tell me about your enterprise plan options?",
        "question": "Is this a cancellation request?",
        "expected_noul_high": False,
    },
    {
        "state": "The product works but the onboarding was really confusing.",
        "question": "Is the customer satisfied?",
        "expected_noul_high": False,  # mixed, leaning no
    },
    {
        "state": "We need this resolved by Friday or we are switching providers.",
        "question": "Is the customer threatening to leave?",
        "expected_noul_high": True,
    },
    {
        "state": "Just checking in to see if there are any updates on my ticket.",
        "question": "Is the customer threatening to leave?",
        "expected_noul_high": False,
    },
    {
        "state": "The API returns a 500 error intermittently on large payloads.",
        "question": "Is this a technical bug?",
        "expected_noul_high": True,
    },
    {
        "state": "What payment methods do you accept?",
        "question": "Is this a technical bug?",
        "expected_noul_high": False,
    },
]


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    print("=" * 70)
    print("PROBE 8: Readout Head Design — Cross-Type Comparison")
    print("=" * 70)

    # --- 8a: Same question as noul, 2-option choice, and 2-level score ---
    print(f"\n--- 8a: Cross-type agreement ({len(CROSS_TYPE_CASES)} cases × 3 types) ---")

    for i, case in enumerate(CROSS_TYPE_CASES):
        state = case["state"]
        question = case["question"]

        # Noul
        t0 = time.monotonic()
        resp_n = client.noul(state, "q", question)
        lat_n = time.monotonic() - t0
        noul_val = resp_n["answers"]["q"]["noul"]

        # Choice (yes/no)
        t0 = time.monotonic()
        resp_c = client.choice(state, "q", question, {"yes": "True, this is the case", "no": "False, this is not the case"})
        lat_c = time.monotonic() - t0
        choice_ans = resp_c["answers"]["q"]

        # Score (2 levels)
        t0 = time.monotonic()
        resp_s = client.score(state, "q", question, ["No, clearly not the case", "Yes, clearly the case"])
        lat_s = time.monotonic() - t0
        score_ans = resp_s["answers"]["q"]

        emit({
            "test": "cross_type",
            "case_idx": i,
            "state": state,
            "question": question,
            "noul": noul_val,
            "noul_latency": round(lat_n, 4),
            "noul_input_tokens": resp_n["usage"]["input_tokens"],
            "noul_output_tokens": resp_n["usage"]["output_tokens"],
            "choice_yes_prob": choice_ans["probabilities"].get("yes", 0),
            "choice_no_prob": choice_ans["probabilities"].get("no", 0),
            "choice_pick": choice_ans["choice"],
            "choice_confidence": choice_ans["confidence"],
            "choice_latency": round(lat_c, 4),
            "choice_input_tokens": resp_c["usage"]["input_tokens"],
            "choice_output_tokens": resp_c["usage"]["output_tokens"],
            "score_val": score_ans["score"],
            "score_prob_0": float(score_ans["probabilities"]["0"]),
            "score_prob_1": float(score_ans["probabilities"]["1"]),
            "score_confidence": score_ans["confidence"],
            "score_latency": round(lat_s, 4),
            "score_input_tokens": resp_s["usage"]["input_tokens"],
            "score_output_tokens": resp_s["usage"]["output_tokens"],
        })

        print(f"  Case {i + 1}: noul={noul_val:.2f}  choice_yes={choice_ans['probabilities'].get('yes', 0):.2f}  score={score_ans['score']:.2f}")

    # --- 8b: Edge cases per type ---
    print(f"\n--- 8b: Edge cases ---")

    edge_cases = [
        ("empty", "", "Is this urgent?"),
        ("whitespace", "   \n\t\n   ", "Is this urgent?"),
        ("single_char", ".", "Is this urgent?"),
        ("numbers_only", "1234567890", "Is this a complaint?"),
        ("all_caps", "EVERYTHING IS FINE NOTHING IS WRONG", "Is there a problem?"),
        ("emoji_only", "😡😡😡🤬🤬🤬", "Is the customer angry?"),
        ("repeated", "refund " * 50, "Is this about a refund?"),
        ("contradiction", "I love your product. I hate your product. I love your product.", "Is the customer satisfied?"),
    ]

    for name, state, question in edge_cases:
        questions = {
            "noul_q": {"type": "noul", "instructions": question},
            "choice_q": {"type": "choice", "instructions": question,
                        "criteria": {"yes": "True", "no": "False"}},
            "score_q": {"type": "score", "instructions": question,
                       "criteria": ["Clearly no", "Clearly yes"]},
        }
        t0 = time.monotonic()
        resp = client.ask(state if state else "x", questions)  # API may reject empty state
        elapsed = time.monotonic() - t0
        a = resp["answers"]

        emit({
            "test": "edge_case",
            "name": name,
            "state": state[:50],
            "noul": a["noul_q"]["noul"],
            "choice_yes": a["choice_q"]["probabilities"].get("yes", 0),
            "choice_pick": a["choice_q"]["choice"],
            "score": a["score_q"]["score"],
            "score_prob_1": float(a["score_q"]["probabilities"]["1"]),
            "elapsed_s": round(elapsed, 4),
        })
        print(f"  {name:15s}: noul={a['noul_q']['noul']:.2f}  choice_yes={a['choice_q']['probabilities'].get('yes', 0):.2f}  score={a['score_q']['score']:.2f}")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    cross = [r for r in records if r["test"] == "cross_type"]

    print("\nCross-type agreement (noul vs choice_yes_prob vs score):")
    diffs_nc = []
    diffs_ns = []
    diffs_cs = []
    for r in cross:
        n = r["noul"]
        c = r["choice_yes_prob"]
        s = r["score_val"]  # 0-1 scale for 2-level score
        diffs_nc.append(abs(n - c))
        diffs_ns.append(abs(n - s))
        diffs_cs.append(abs(c - s))

    print(f"  Mean |noul - choice_yes|: {sum(diffs_nc)/len(diffs_nc):.4f}")
    print(f"  Mean |noul - score|:      {sum(diffs_ns)/len(diffs_ns):.4f}")
    print(f"  Mean |choice_yes - score|:{sum(diffs_cs)/len(diffs_cs):.4f}")
    print(f"  Max  |noul - choice_yes|: {max(diffs_nc):.4f}")

    # Agreement on binary decision
    agree_all = 0
    for r in cross:
        n_yes = r["noul"] > 0.5
        c_yes = r["choice_yes_prob"] > 0.5
        s_yes = r["score_val"] > 0.5
        if n_yes == c_yes == s_yes:
            agree_all += 1
    print(f"\n  All 3 types agree on binary decision: {agree_all}/{len(cross)} ({100*agree_all/len(cross):.0f}%)")

    # Latency comparison
    print("\nLatency by type (median):")
    for key, label in [("noul_latency", "noul"), ("choice_latency", "choice"), ("score_latency", "score")]:
        lats = sorted(r[key] for r in cross)
        median = lats[len(lats) // 2]
        print(f"  {label:8s}: {median:.3f}s")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
