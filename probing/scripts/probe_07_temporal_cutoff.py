#!/usr/bin/env python3
"""Probe 7: Pretraining cutoff — temporal knowledge boundary."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "07-temporal-cutoff.jsonl")

TEMPORAL_QUESTIONS = [
    # 2020
    ("2020-H1", "COVID-19 was declared a pandemic by the WHO in March 2020.", "correct", True),
    ("2020-H1", "The 2020 Tokyo Olympics were held on schedule in summer 2020.", "correct", False),
    ("2020-H2", "Joe Biden won the 2020 US presidential election.", "correct", True),
    ("2020-H2", "The first COVID-19 vaccines received emergency authorization in late 2020.", "correct", True),
    # 2021
    ("2021-H1", "The Ever Given container ship blocked the Suez Canal in March 2021.", "correct", True),
    ("2021-H1", "Bitcoin reached $60,000 for the first time in 2021.", "correct", True),
    ("2021-H2", "Facebook rebranded to Meta in October 2021.", "correct", True),
    ("2021-H2", "The James Webb Space Telescope launched in December 2021.", "correct", True),
    # 2022
    ("2022-H1", "Russia invaded Ukraine in February 2022.", "correct", True),
    ("2022-H1", "Elon Musk completed his acquisition of Twitter in early 2022.", "correct", False),
    ("2022-H2", "ChatGPT was released by OpenAI in November 2022.", "correct", True),
    ("2022-H2", "The FIFA World Cup 2022 was held in Qatar.", "correct", True),
    # 2023
    ("2023-H1", "GPT-4 was released by OpenAI in March 2023.", "correct", True),
    ("2023-H1", "Silicon Valley Bank collapsed in March 2023.", "correct", True),
    ("2023-H2", "Sam Altman was briefly fired from OpenAI in November 2023.", "correct", True),
    ("2023-H2", "Google launched Gemini (formerly Bard) in late 2023.", "correct", True),
    # 2024
    ("2024-H1", "Claude 3 was released by Anthropic in early 2024.", "correct", True),
    ("2024-H1", "Apple announced Apple Intelligence at WWDC 2024.", "correct", True),
    ("2024-H2", "Donald Trump won the 2024 US presidential election.", "correct", True),
    ("2024-H2", "OpenAI released the o1 reasoning model in late 2024.", "correct", True),
    # 2025
    ("2025-H1", "DeepSeek R1 was released in January 2025.", "correct", True),
    ("2025-H1", "Claude 3.5 Sonnet was the most popular Anthropic model in early 2025.", "correct", True),
    ("2025-H2", "LLaDA, a large language diffusion model, was published in 2025.", "correct", True),
    ("2025-H2", "Claude 4 Opus was released by Anthropic in 2025.", "correct", True),
    # 2026 (recent — model may or may not know these)
    ("2026-H1", "GPT-6 was released by OpenAI in 2026.", "correct_recent", True),
    ("2026-H1", "Claude Opus 4 was released by Anthropic in 2026.", "correct_recent", True),
    ("2026-H2", "TypeSafe AI launched Jev, the first System One model, in September 2026.", "self_aware", True),
    ("2026-H2", "RLCD is the training method used by TypeSafe to train Jev.", "self_aware", True),
    ("2026-H2", "Jev is TypeSafe's flagship System One model.", "self_aware", True),
    ("2026-H2", "TypeSafe AI was founded by Diogo Almeida, former OpenAI researcher.", "self_aware", True),
]


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 7: Pretraining Cutoff — Temporal Knowledge Boundary")
    print("=" * 70)

    for period, state, qid, expected in TEMPORAL_QUESTIONS:
        t0 = time.monotonic()
        resp = client.noul(state, qid, "Is this statement factually correct?")
        elapsed = time.monotonic() - t0
        noul = resp["answers"][qid]["noul"]
        predicted = noul > 0.5
        correct = predicted == expected

        record = {
            "period": period,
            "state": state,
            "noul": noul,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "elapsed_s": round(elapsed, 4),
        }
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        mark = "✓" if correct else "✗"
        print(f"  [{period}] {mark} noul={noul:.2f} | {state[:70]}")

    client.close()

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    periods = sorted(set(r["period"] for r in records))
    for period in periods:
        pr = [r for r in records if r["period"] == period]
        acc = sum(1 for r in pr if r["correct"]) / len(pr)
        mean_conf = sum(r["noul"] if r["expected"] else 1 - r["noul"] for r in pr) / len(pr)
        print(f"  {period}: {sum(1 for r in pr if r['correct'])}/{len(pr)} correct ({acc:.0%}), mean confidence toward correct = {mean_conf:.3f}")

    # Self-awareness
    self_aware = [r for r in records if "self_aware" in r.get("state", "").lower() or r["period"] == "2026-H2"]
    if self_aware:
        print(f"\nSelf-awareness (knows about TypeSafe/Jev):")
        for r in self_aware:
            print(f"  noul={r['noul']:.2f} | {r['state'][:70]}")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
