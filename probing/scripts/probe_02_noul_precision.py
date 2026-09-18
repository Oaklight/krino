#!/usr/bin/env python3
"""Probe 2: Noul precision quantization — output head fingerprint.

Collects 500+ noul/choice/score values and analyzes numerical distribution
to identify quantization patterns in the output head.
"""

import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "02-noul-precision.jsonl")

STATES = [
    "My order arrived damaged and I want a refund.",
    "Thanks for the quick delivery! Everything looks great.",
    "I'm not sure if this is the right product for me.",
    "YOUR SERVICE IS TERRIBLE AND I WANT MY MONEY BACK",
    "Could you tell me your business hours?",
    "The app crashes every time I try to upload a photo on iPhone.",
    "I noticed a minor typo on your about page.",
    "We are extremely satisfied with your service. Best purchase ever!",
    "I've been trying to connect my Stripe account for 3 days and it keeps failing.",
    "The product is okay, nothing special but it works.",
    "I ordered size 10 but received size 8. Please fix this ASAP.",
    "Just wanted to say your customer support team is amazing!",
    "My subscription was cancelled without warning.",
    "Can I get a discount on bulk orders?",
    "The download link in your email is broken.",
    "I accidentally placed a duplicate order. Can you cancel one?",
    "Your competitor offers the same thing for half the price.",
    "The installation guide is missing step 3.",
    "I've been a loyal customer for 5 years and this is how you treat me?",
    "Hello, I would like to inquire about your return policy.",
    "客户需要退款，订单号12345。",
    "Ce produit est défectueux, je souhaite un remboursement.",
    "このサービスにとても満足しています。",
    "Der Versand war sehr schnell, danke!",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
    "Patient presents with acute myocardial infarction, STEMI protocol initiated.",
    "The yield curve inverted last Tuesday, suggesting recession risk.",
    "BREAKING: Server is down, all hands on deck!",
    "I think there might be a small issue with my account, but I'm not sure.",
    "🔥🔥🔥 This product is absolutely amazing! 10/10 would recommend! 🔥🔥🔥",
    "The weather is nice today.",
    "Please process my refund for order #789. I've asked three times already.",
    "I'm considering switching to a competitor.",
    "The new feature you added last week is really helpful.",
    "My credit card was charged twice for the same order.",
    "Can someone explain how to use the API?",
    "I need this resolved by end of day or I'm cancelling my subscription.",
    "Just browsing, no issues.",
    "The software worked perfectly for months but now it freezes constantly.",
    "Thank you for resolving my issue so quickly last time!",
    "I want to upgrade my plan but the button doesn't work.",
    "Is this product compatible with Windows 11?",
    "Nobody on our team can log in since this morning.",
    "The color in the photo doesn't match what I received.",
    "I'd like to schedule a demo with your sales team.",
    "This is urgent - our production system is down.",
    "How do I export my data from your platform?",
    "The mobile app is much slower than the desktop version.",
    "I love the new design update!",
]

NOUL_QUESTIONS = [
    ("is_urgent", "Does this message express urgency?"),
    ("is_refund", "Is the customer requesting a refund?"),
    ("is_angry", "Is the customer angry or frustrated?"),
    ("is_positive", "Is the overall sentiment positive?"),
    ("needs_action", "Does this require immediate action from support?"),
    ("is_technical", "Is this a technical issue?"),
    ("is_complaint", "Is this a complaint?"),
    ("mentions_competitor", "Does this mention a competitor or alternative?"),
    ("is_repeat", "Has the customer contacted support about this before?"),
    ("is_escalation", "Should this be escalated to a manager?"),
]

CHOICE_QUESTIONS = [
    ("department", "Which team should handle this?",
     {"billing": "Payment issues", "shipping": "Delivery", "returns": "Returns", "tech": "Technical support", "sales": "Sales inquiries"}),
    ("tone", "What is the customer's tone?",
     {"calm": None, "polite": None, "frustrated": None, "angry": None}),
]

SCORE_QUESTIONS = [
    ("frustration", "How frustrated is the customer?",
     ["Calm, just stating facts", "Mildly annoyed", "Frustrated but civil", "Very frustrated", "Furious, threatening"]),
    ("severity", "How severe is the reported issue?",
     ["No issue", "Minor inconvenience", "Moderate problem", "Major issue", "Critical, system down"]),
]


def main():
    client = JevClient()
    all_nouls = []
    all_choice_probs = []
    all_score_probs = []
    record_count = 0

    print("=" * 70)
    print("PROBE 2: Noul Precision Quantization")
    print(f"States: {len(STATES)}, Noul questions: {len(NOUL_QUESTIONS)}")
    print(f"Expected data points: ~{len(STATES) * len(NOUL_QUESTIONS)} nouls")
    print("=" * 70)

    with open(OUTFILE, "w") as f:
        # Batch 5 noul questions per call to reduce API calls
        for i, state in enumerate(STATES):
            # Noul questions in batches of 5
            for batch_start in range(0, len(NOUL_QUESTIONS), 5):
                batch = NOUL_QUESTIONS[batch_start:batch_start + 5]
                questions = {qid: {"type": "noul", "instructions": instr} for qid, instr in batch}

                # Add one choice and one score question to each batch for bonus data
                if batch_start == 0:
                    cid, cinstr, ccriteria = CHOICE_QUESTIONS[0]
                    questions[cid] = {"type": "choice", "instructions": cinstr, "criteria": ccriteria}
                    sid, sinstr, scriteria = SCORE_QUESTIONS[0]
                    questions[sid] = {"type": "score", "instructions": sinstr, "criteria": scriteria}
                elif batch_start == 5:
                    cid, cinstr, ccriteria = CHOICE_QUESTIONS[1]
                    questions[cid] = {"type": "choice", "instructions": cinstr, "criteria": ccriteria}
                    sid, sinstr, scriteria = SCORE_QUESTIONS[1]
                    questions[sid] = {"type": "score", "instructions": sinstr, "criteria": scriteria}

                t0 = time.monotonic()
                resp = client.ask(state, questions)
                elapsed = time.monotonic() - t0

                for qid, answer in resp["answers"].items():
                    record = {
                        "state_idx": i,
                        "question_id": qid,
                        "type": answer["type"],
                        "elapsed_s": round(elapsed, 4),
                    }
                    if answer["type"] == "noul":
                        record["noul"] = answer["noul"]
                        all_nouls.append(answer["noul"])
                    elif answer["type"] == "choice":
                        record["choice"] = answer["choice"]
                        record["probabilities"] = answer["probabilities"]
                        record["confidence"] = answer["confidence"]
                        all_choice_probs.extend(answer["probabilities"].values())
                    elif answer["type"] == "score":
                        record["score"] = answer["score"]
                        record["probabilities"] = answer["probabilities"]
                        record["confidence"] = answer["confidence"]
                        all_score_probs.extend(float(v) for v in answer["probabilities"].values())

                    f.write(json.dumps(record) + "\n")
                    record_count += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(STATES)} states ({record_count} records)")

    client.close()

    print(f"\n{'=' * 70}")
    print(f"Done. {record_count} records written to {OUTFILE}")
    print(f"  Noul values: {len(all_nouls)}")
    print(f"  Choice probability values: {len(all_choice_probs)}")
    print(f"  Score probability values: {len(all_score_probs)}")

    # --- Precision Analysis ---
    print(f"\n{'=' * 70}")
    print("PRECISION ANALYSIS")
    print("=" * 70)

    def analyze_precision(values, name):
        if not values:
            return
        print(f"\n--- {name} ({len(values)} values) ---")

        # Decimal places
        decimal_places = []
        for v in values:
            s = f"{v:.15g}"
            if "." in s:
                decimal_places.append(len(s.split(".")[1].rstrip("0")) if s.split(".")[1].rstrip("0") else 0)
            else:
                decimal_places.append(0)
        dp_counts = Counter(decimal_places)
        print(f"Decimal places distribution: {dict(sorted(dp_counts.items()))}")

        # Unique values
        unique = sorted(set(values))
        print(f"Unique values: {len(unique)} out of {len(values)}")
        if len(unique) <= 30:
            print(f"All unique: {unique}")

        # Check for quantization grids
        for grid_size in [64, 100, 128, 200, 256, 500, 512, 1000, 1024, 2048, 4096]:
            residuals = [abs(v * grid_size - round(v * grid_size)) for v in values if 0 < v < 1]
            if residuals:
                max_residual = max(residuals)
                mean_residual = sum(residuals) / len(residuals)
                if max_residual < 0.01:
                    print(f"  *** MATCHES 1/{grid_size} grid (max residual: {max_residual:.6f}) ***")
                elif mean_residual < 0.01:
                    print(f"  Near 1/{grid_size} grid (mean residual: {mean_residual:.6f}, max: {max_residual:.6f})")

        # Min/max nonzero
        nonzero = [v for v in values if v > 0]
        if nonzero:
            print(f"Min nonzero: {min(nonzero)}")
            print(f"Max: {max(values)}")

        # Gaps between sorted unique values
        if len(unique) >= 3:
            gaps = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
            nonzero_gaps = [g for g in gaps if g > 1e-10]
            if nonzero_gaps:
                print(f"Min gap: {min(nonzero_gaps):.6f}")
                print(f"Max gap: {max(nonzero_gaps):.6f}")
                print(f"Median gap: {sorted(nonzero_gaps)[len(nonzero_gaps) // 2]:.6f}")

    analyze_precision(all_nouls, "Noul values")
    analyze_precision(all_choice_probs, "Choice probability values")
    analyze_precision(all_score_probs, "Score probability values")


if __name__ == "__main__":
    main()
