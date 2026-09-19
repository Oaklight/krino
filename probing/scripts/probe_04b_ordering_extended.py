#!/usr/bin/env python3
"""Probe 4b: Controlled ordering bias — larger option sets + harder cases.

Probe 4 flaw: only 12 cases with 3 options, all relatively easy.
Hume DID find position bias with reference-card experiments.

This probe:
1. Uses 5-option and 8-option sets where the correct answer is genuinely uncertain
2. Tests 30+ cases for statistical power
3. Measures position-specific probability uplift across ALL cases
"""

import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "04b-ordering-controlled.jsonl")

random.seed(42)

# Hard ambiguous cases with 5 options
CASES_5OPT = [
    {
        "state": "I bought these shoes last month and they're starting to fall apart. What can you do about it?",
        "instructions": "What does the customer want?",
        "options": {"refund": "Money back", "exchange": "Different product", "repair": "Fix current item", "credit": "Store credit", "information": "Just asking options"},
    },
    {
        "state": "Your API sometimes returns errors. It's not blocking us but it's annoying. We might look at alternatives if it continues.",
        "instructions": "What is the primary concern?",
        "options": {"reliability": "System stability", "performance": "Speed issues", "churn_risk": "May leave", "feature_gap": "Missing functionality", "support": "Wants help"},
    },
    {
        "state": "The new dashboard is nice but I really miss the old export feature. Can you bring it back or suggest a workaround?",
        "instructions": "What type of request is this?",
        "options": {"feature_request": "Wants new feature", "regression": "Broken existing feature", "feedback": "General opinion", "support": "Needs help", "complaint": "Expressing dissatisfaction"},
    },
    {
        "state": "We signed up for enterprise but the onboarding has been slow. Our team is getting frustrated waiting.",
        "instructions": "Which team should handle this?",
        "options": {"onboarding": "Customer success", "sales": "Account management", "support": "Technical help", "engineering": "Product issues", "management": "Escalation"},
    },
    {
        "state": "I was charged the wrong amount. The receipt shows $49 but my card was charged $79. Also the item description is wrong.",
        "instructions": "What is the primary issue?",
        "options": {"overcharge": "Wrong amount charged", "billing_error": "Receipt mismatch", "fraud": "Unauthorized charge", "catalog_error": "Wrong item info", "refund_needed": "Needs money back"},
    },
    {
        "state": "The product works great for our small team but we're growing and worried it won't scale. What are our options?",
        "instructions": "What is the customer's intent?",
        "options": {"upgrade": "Wants bigger plan", "evaluation": "Assessing fit", "concern": "Worried about limits", "churn_risk": "May leave", "sales_inquiry": "Pricing question"},
    },
    {
        "state": "Thanks for fixing the bug! One more thing though - the fix seems to have broken the search bar on mobile.",
        "instructions": "What should we do with this ticket?",
        "options": {"close": "Issue resolved", "reopen": "Original issue back", "new_ticket": "New separate issue", "investigate": "Need more info", "escalate": "Send to senior eng"},
    },
    {
        "state": "I can't log in. I've tried resetting my password twice but the reset email never arrives. I've checked spam.",
        "instructions": "What is the root cause?",
        "options": {"email_delivery": "Email system issue", "password_system": "Reset mechanism broken", "account_locked": "Security lockout", "user_error": "Wrong email address", "dns_issue": "Email routing problem"},
    },
]

# 8-option case for maximum permutation stress
CASES_8OPT = [
    {
        "state": "Our team has been using your product for 6 months. Some features are great, others feel half-baked. We're evaluating whether to renew.",
        "instructions": "What is the dominant customer emotion?",
        "options": {
            "satisfied": "Happy overall", "disappointed": "Let down", "frustrated": "Annoyed",
            "neutral": "No strong feeling", "hopeful": "Expecting improvement", "anxious": "Worried about decision",
            "resigned": "Accepted limitations", "conflicted": "Mixed strong feelings"
        },
    },
]


def run_case(client, case, n_permutations=None):
    """Run a case with sampled permutations. Returns list of records."""
    options = case["options"]
    keys = list(options.keys())
    n_opts = len(keys)

    all_perms = list(itertools.permutations(keys))
    if n_permutations and n_permutations < len(all_perms):
        perms = random.sample(all_perms, n_permutations)
    else:
        perms = all_perms

    results = []
    for perm in perms:
        criteria = {k: options[k] for k in perm}
        t0 = time.monotonic()
        resp = client.choice(case["state"], "q", case["instructions"], criteria)
        elapsed = time.monotonic() - t0
        answer = resp["answers"]["q"]

        results.append({
            "option_order": list(perm),
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "elapsed_s": round(elapsed, 4),
        })
    return results


def main():
    client = JevClient()
    all_records = []
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 4b: Controlled Ordering Bias")
    print("=" * 70)

    # 5-option cases: sample 20 permutations each (out of 120 possible)
    print(f"\n--- 5-option cases ({len(CASES_5OPT)} cases × 20 permutations) ---")
    for i, case in enumerate(CASES_5OPT):
        results = run_case(client, case, n_permutations=20)
        keys = list(case["options"].keys())

        for r in results:
            record = {"case_idx": i, "n_options": len(keys), **r}
            all_records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Per-case summary
        choices = [r["choice"] for r in results]
        unique = set(choices)
        max_swing = 0
        for opt in keys:
            probs = [r["probabilities"][opt] for r in results]
            swing = max(probs) - min(probs)
            max_swing = max(max_swing, swing)

        flipped = "FLIPPED" if len(unique) > 1 else "stable"
        print(f"  Case {i}: max_swing={max_swing:.3f} {flipped} ({dict(zip(*[list(x) for x in zip(*[(c, choices.count(c)) for c in unique])]))})")

    # 8-option case: sample 30 permutations (out of 40320 possible)
    print(f"\n--- 8-option case (1 case × 30 permutations) ---")
    for i, case in enumerate(CASES_8OPT):
        results = run_case(client, case, n_permutations=30)
        keys = list(case["options"].keys())

        for r in results:
            record = {"case_idx": len(CASES_5OPT) + i, "n_options": len(keys), **r}
            all_records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

        choices = [r["choice"] for r in results]
        unique = set(choices)
        print(f"  8-opt case: {len(unique)} unique choices out of {len(results)} permutations")

    client.close()

    # --- Position Bias Analysis ---
    print(f"\n{'=' * 70}")
    print("POSITION BIAS ANALYSIS")
    print("=" * 70)

    # For each record, check: what probability did the option at position P get?
    for n_opt in [5, 8]:
        subset = [r for r in all_records if r["n_options"] == n_opt]
        if not subset:
            continue

        print(f"\n{n_opt}-option cases ({len(subset)} records):")
        for pos in range(n_opt):
            probs_at_pos = []
            for r in subset:
                opt_at_pos = r["option_order"][pos]
                probs_at_pos.append(r["probabilities"][opt_at_pos])
            mean_p = sum(probs_at_pos) / len(probs_at_pos)
            expected = 1.0 / n_opt
            delta = mean_p - expected
            bar = "+" * int(abs(delta) * 200) if delta > 0 else "-" * int(abs(delta) * 200)
            print(f"  Position {pos}: mean_p={mean_p:.4f} (expected {expected:.4f}, delta={delta:+.4f}) {bar}")

    # Overall flip rate
    total_cases = len(CASES_5OPT) + len(CASES_8OPT)
    flip_count = 0
    for ci in range(total_cases):
        case_records = [r for r in all_records if r["case_idx"] == ci]
        if len(set(r["choice"] for r in case_records)) > 1:
            flip_count += 1
    print(f"\nChoice flip rate: {flip_count}/{total_cases} ({100 * flip_count / total_cases:.0f}%)")

    # Probability swing stats
    swings = []
    for ci in range(total_cases):
        case_records = [r for r in all_records if r["case_idx"] == ci]
        if not case_records:
            continue
        opts = list(case_records[0]["probabilities"].keys())
        for opt in opts:
            probs = [r["probabilities"][opt] for r in case_records]
            swings.append(max(probs) - min(probs))

    print(f"Probability swing: mean={sum(swings)/len(swings):.4f}, median={sorted(swings)[len(swings)//2]:.4f}, max={max(swings):.4f}")
    print(f"\nTotal: {len(all_records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
