#!/usr/bin/env python3
"""Probe 4: Option ordering bias on ambiguous cases.

Tests all permutations of choice options on genuinely ambiguous inputs
where probabilities should be close to 50/50.
"""

import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "04-ordering-bias.jsonl")

# Ambiguous cases: designed so the answer is NOT obvious
AMBIGUOUS_CASES = [
    {
        "state": "I bought these shoes last month and they're starting to fall apart. What can you do?",
        "instructions": "What does the customer want?",
        "options": {"refund": "Money back", "exchange": "Replace with new pair", "repair": "Fix the current item"},
    },
    {
        "state": "This feature doesn't work the way I expected. It would be great if it could also handle CSV files.",
        "instructions": "What type of message is this?",
        "options": {"bug_report": "Reporting broken functionality", "feature_request": "Asking for new capability", "feedback": "General product feedback"},
    },
    {
        "state": "I've been a customer for years and I'm honestly disappointed with the recent changes.",
        "instructions": "What is the primary emotion?",
        "options": {"sadness": "Feeling let down or sad", "anger": "Feeling annoyed or angry", "disappointment": "Feeling unmet expectations"},
    },
    {
        "state": "The API sometimes returns a 500 error. It seems to happen about once an hour.",
        "instructions": "How severe is this issue?",
        "options": {"low": "Minor, intermittent issue", "medium": "Significant but workaround exists", "high": "Blocking productivity"},
    },
    {
        "state": "Can you help me understand the pricing for teams? We might be interested in upgrading.",
        "instructions": "Which team handles this?",
        "options": {"sales": "Revenue and pricing", "support": "Customer help", "billing": "Payment and invoices"},
    },
    {
        "state": "The dashboard loads but the charts are empty. I can still see the data in the table view.",
        "instructions": "What type of issue is this?",
        "options": {"frontend_bug": "UI rendering problem", "data_issue": "Data not loading correctly", "configuration": "Settings need adjustment"},
    },
    {
        "state": "I signed up for the free trial but I'm already being charged.",
        "instructions": "Which department should handle this?",
        "options": {"billing": "Payment disputes", "sales": "Trial and plan management", "legal": "Compliance and disputes"},
    },
    {
        "state": "Your product works great for small files but chokes on anything over 1GB.",
        "instructions": "What is this message about?",
        "options": {"performance": "Speed and scaling issues", "limitation": "Product capability constraints", "bug": "Unexpected broken behavior"},
    },
    {
        "state": "We need to migrate our data from your platform. Can you provide an export?",
        "instructions": "What is the customer's intent?",
        "options": {"churn": "Planning to leave", "data_management": "Routine data operations", "compliance": "Regulatory data requirements"},
    },
    {
        "state": "The onboarding tutorial was confusing but after I figured it out the product is solid.",
        "instructions": "What is the overall sentiment?",
        "options": {"positive": "Generally happy", "negative": "Generally unhappy", "mixed": "Both positive and negative elements"},
    },
    {
        "state": "I tried to reset my password but the email never arrived. I checked spam.",
        "instructions": "What is the root cause?",
        "options": {"email_deliverability": "Email system issue", "account_issue": "Account configuration problem", "user_error": "Customer made a mistake"},
    },
    {
        "state": "The product does what it says but I wish it had better documentation.",
        "instructions": "How satisfied is the customer?",
        "options": {"satisfied": "Happy with the product", "neutral": "Neither happy nor unhappy", "dissatisfied": "Unhappy with the experience"},
    },
]


def main():
    client = JevClient()
    records = []

    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 4: Option Ordering Bias (Ambiguous Cases)")
    print(f"Cases: {len(AMBIGUOUS_CASES)}, each with all permutations")
    print("=" * 70)

    for case_idx, case in enumerate(AMBIGUOUS_CASES):
        option_names = list(case["options"].keys())
        perms = list(itertools.permutations(option_names))
        print(f"\n  Case {case_idx + 1}: {len(option_names)} options → {len(perms)} permutations")
        print(f"  State: {case['state'][:60]}...")

        case_results = []
        for perm_idx, perm in enumerate(perms):
            criteria = {k: case["options"][k] for k in perm}
            t0 = time.monotonic()
            resp = client.choice(case["state"], "q", case["instructions"], criteria)
            elapsed = time.monotonic() - t0
            answer = resp["answers"]["q"]

            record = {
                "case_idx": case_idx,
                "perm_idx": perm_idx,
                "option_order": list(perm),
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer["confidence"],
                "elapsed_s": round(elapsed, 4),
            }
            records.append(record)
            case_results.append(record)

            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Per-case analysis
        choices = [r["choice"] for r in case_results]
        unique_choices = set(choices)
        choice_flips = len(unique_choices) > 1

        # Probability swings per option
        for opt in option_names:
            probs = [r["probabilities"][opt] for r in case_results]
            swing = max(probs) - min(probs)
            print(f"    {opt:20s}: min={min(probs):.4f} max={max(probs):.4f} swing={swing:.4f}")

        if choice_flips:
            from collections import Counter
            print(f"    *** CHOICE FLIPPED: {dict(Counter(choices))} ***")
        else:
            print(f"    Choice stable: always '{choices[0]}'")

    client.close()

    # --- Aggregate Analysis ---
    print(f"\n{'=' * 70}")
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    total_cases = len(AMBIGUOUS_CASES)
    flip_count = 0
    all_swings = []

    for case_idx in range(total_cases):
        case_records = [r for r in records if r["case_idx"] == case_idx]
        choices = [r["choice"] for r in case_records]
        if len(set(choices)) > 1:
            flip_count += 1

        option_names = list(AMBIGUOUS_CASES[case_idx]["options"].keys())
        for opt in option_names:
            probs = [r["probabilities"][opt] for r in case_records]
            all_swings.append(max(probs) - min(probs))

    print(f"Cases with choice flips: {flip_count}/{total_cases} ({100 * flip_count / total_cases:.1f}%)")
    print(f"Mean probability swing: {sum(all_swings) / len(all_swings):.4f}")
    print(f"Max probability swing: {max(all_swings):.4f}")
    print(f"Median probability swing: {sorted(all_swings)[len(all_swings) // 2]:.4f}")

    # Position bias analysis
    print("\n--- Position Bias (across all cases) ---")
    for pos in range(max(len(c["options"]) for c in AMBIGUOUS_CASES)):
        probs_at_pos = []
        for r in records:
            option_at_pos = r["option_order"][pos] if pos < len(r["option_order"]) else None
            if option_at_pos:
                probs_at_pos.append(r["probabilities"][option_at_pos])
        if probs_at_pos:
            mean_p = sum(probs_at_pos) / len(probs_at_pos)
            print(f"  Position {pos}: mean probability of option at this position = {mean_p:.4f} (n={len(probs_at_pos)})")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
