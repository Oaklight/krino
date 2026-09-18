#!/usr/bin/env python3
"""Probe 3c: Rigorous bidirectional test.

The flaw in Probe 3b: in causal attention with KV caching, the question suffix
attends to ALL state KVs. A strong model can compose information from any positions
at question-level attention, even if individual state token KVs only saw left context.

This probe makes composition STRUCTURALLY HARDER for causal models by:
1. Using made-up codes where the DEFINITION comes AFTER the usage
2. Scaling the number of codes (N=1..10) so the question must do N-way cross-referencing
3. Causal prediction: degradation increases with N (harder to compose at query time)
   Bidirectional prediction: flat performance regardless of N (definitions baked into KVs)
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "03c-bidi-rigorous.jsonl")

random.seed(42)

CODES = [
    ("ZX7", "urgent"),
    ("QM3", "billing"),
    ("KP9", "refund"),
    ("VN2", "shipping"),
    ("WT5", "complaint"),
    ("HJ8", "technical"),
    ("BF4", "escalation"),
    ("RL6", "positive"),
    ("DS1", "cancellation"),
    ("YC0", "duplicate"),
]

DISTRACTORS = [
    "The system processed the request normally.",
    "No additional context was provided by the user.",
    "The ticket was created via the web portal.",
    "Standard processing time applies.",
    "The account is in good standing.",
]


def build_state_def_first(n_codes: int, target_idx: int) -> tuple[str, str, str, bool]:
    """Definition BEFORE usage. Causal model should handle this fine."""
    codes = CODES[:n_codes]
    target_code, target_meaning = codes[target_idx]

    parts = []
    # Definitions first
    parts.append("Code definitions:")
    for code, meaning in codes:
        parts.append(f"  {code} = {meaning}")
    # Then distractor text
    parts.append("")
    for d in random.sample(DISTRACTORS, min(3, len(DISTRACTORS))):
        parts.append(d)
    # Then usage
    parts.append("")
    parts.append(f"Customer ticket is tagged: {target_code}")

    state = "\n".join(parts)
    question = f"Is this ticket about {target_meaning}?"
    return state, "q", question, True


def build_state_def_last(n_codes: int, target_idx: int) -> tuple[str, str, str, bool]:
    """Definition AFTER usage. Causal model must compose at question level."""
    codes = CODES[:n_codes]
    target_code, target_meaning = codes[target_idx]

    parts = []
    # Usage first
    parts.append(f"Customer ticket is tagged: {target_code}")
    parts.append("")
    # Then distractor text
    for d in random.sample(DISTRACTORS, min(3, len(DISTRACTORS))):
        parts.append(d)
    # Definitions last
    parts.append("")
    parts.append("Code definitions:")
    for code, meaning in codes:
        parts.append(f"  {code} = {meaning}")

    state = "\n".join(parts)
    question = f"Is this ticket about {target_meaning}?"
    return state, "q", question, True


def build_state_wrong_code(n_codes: int, target_idx: int) -> tuple[str, str, str, bool]:
    """Ask about a DIFFERENT code's meaning. Should answer False."""
    codes = CODES[:n_codes]
    target_code, _ = codes[target_idx]
    wrong_idx = (target_idx + 1) % n_codes
    _, wrong_meaning = codes[wrong_idx]

    parts = []
    parts.append(f"Customer ticket is tagged: {target_code}")
    parts.append("")
    for d in random.sample(DISTRACTORS, min(3, len(DISTRACTORS))):
        parts.append(d)
    parts.append("")
    parts.append("Code definitions:")
    for code, meaning in codes:
        parts.append(f"  {code} = {meaning}")

    state = "\n".join(parts)
    question = f"Is this ticket about {wrong_meaning}?"
    return state, "q", question, False


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    print("=" * 70)
    print("PROBE 3c: Rigorous Bidirectional Test (Scaling Code Definitions)")
    print("=" * 70)

    # Test with N=1,2,3,5,7,10 codes
    for n_codes in [1, 2, 3, 5, 7, 10]:
        print(f"\n--- N={n_codes} codes ---")
        for target_idx in range(min(n_codes, 3)):  # test up to 3 targets per N
            for placement, builder in [("def_first", build_state_def_first),
                                        ("def_last", build_state_def_last),
                                        ("wrong_code", build_state_wrong_code)]:
                state, qid, question, expected = builder(n_codes, target_idx)
                t0 = time.monotonic()
                resp = client.noul(state, qid, question)
                elapsed = time.monotonic() - t0
                noul = resp["answers"][qid]["noul"]
                predicted = noul > 0.5
                correct = predicted == expected

                record = {
                    "n_codes": n_codes,
                    "target_idx": target_idx,
                    "placement": placement,
                    "expected": expected,
                    "noul": noul,
                    "predicted": predicted,
                    "correct": correct,
                    "input_tokens": resp["usage"]["input_tokens"],
                    "elapsed_s": round(elapsed, 4),
                }
                emit(record)

        # Summary for this N
        def_first_r = [r for r in records if r["n_codes"] == n_codes and r["placement"] == "def_first"]
        def_last_r = [r for r in records if r["n_codes"] == n_codes and r["placement"] == "def_last"]
        wrong_r = [r for r in records if r["n_codes"] == n_codes and r["placement"] == "wrong_code"]

        f_acc = sum(1 for r in def_first_r if r["correct"]) / len(def_first_r) if def_first_r else 0
        l_acc = sum(1 for r in def_last_r if r["correct"]) / len(def_last_r) if def_last_r else 0
        w_acc = sum(1 for r in wrong_r if r["correct"]) / len(wrong_r) if wrong_r else 0
        f_noul = sum(r["noul"] for r in def_first_r) / len(def_first_r) if def_first_r else 0
        l_noul = sum(r["noul"] for r in def_last_r) / len(def_last_r) if def_last_r else 0

        print(f"  def_first: acc={f_acc:.0%} mean_noul={f_noul:.3f}")
        print(f"  def_last:  acc={l_acc:.0%} mean_noul={l_noul:.3f}  delta={l_noul - f_noul:+.3f}")
        print(f"  wrong:     acc={w_acc:.0%}")

    client.close()

    # --- Key Analysis ---
    print(f"\n{'=' * 70}")
    print("KEY ANALYSIS: Does def_last degrade as N increases?")
    print("=" * 70)
    print("Causal prediction: def_last accuracy/noul DROPS as N grows")
    print("Bidirectional prediction: def_last stays FLAT regardless of N")
    print()

    print(f"{'N':>3s} | {'def_first noul':>14s} | {'def_last noul':>13s} | {'delta':>7s} | {'def_first acc':>13s} | {'def_last acc':>12s}")
    print("-" * 75)
    for n in sorted(set(r["n_codes"] for r in records)):
        df = [r for r in records if r["n_codes"] == n and r["placement"] == "def_first"]
        dl = [r for r in records if r["n_codes"] == n and r["placement"] == "def_last"]
        fn = sum(r["noul"] for r in df) / len(df) if df else 0
        ln = sum(r["noul"] for r in dl) / len(dl) if dl else 0
        fa = sum(1 for r in df if r["correct"]) / len(df) if df else 0
        la = sum(1 for r in dl if r["correct"]) / len(dl) if dl else 0
        print(f"{n:3d} | {fn:14.3f} | {ln:13.3f} | {ln - fn:+7.3f} | {fa:13.0%} | {la:12.0%}")

    # Trend analysis
    ns = sorted(set(r["n_codes"] for r in records))
    last_nouls = []
    for n in ns:
        dl = [r for r in records if r["n_codes"] == n and r["placement"] == "def_last"]
        last_nouls.append(sum(r["noul"] for r in dl) / len(dl) if dl else 0)

    if len(last_nouls) >= 2:
        trend = last_nouls[-1] - last_nouls[0]
        print(f"\ndef_last noul trend (N={ns[0]} → N={ns[-1]}): {trend:+.3f}")
        if trend < -0.10:
            print("→ SIGNIFICANT DEGRADATION: consistent with CAUSAL model")
        elif abs(trend) < 0.05:
            print("→ FLAT: consistent with BIDIRECTIONAL model")
        else:
            print("→ INCONCLUSIVE")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
