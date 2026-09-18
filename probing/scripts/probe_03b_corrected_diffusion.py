#!/usr/bin/env python3
"""Probe 3b: Corrected diffusion test.

Fixes Probe 3's flaw (character-level masking breaks BPE tokens regardless of architecture).
Uses word-level masking, bidirectional context placement, and token-level noise insertion.
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
OUTFILE = os.path.join(RESULTS_DIR, "03b-corrected-diffusion.jsonl")

random.seed(42)

# --- 3b-1: Word-level masking ---
WORD_MASKING_CASES = [
    {
        "state": "The customer is requesting a refund for their damaged order.",
        "qid": "is_refund", "instructions": "Is the customer requesting a refund?",
        "key_words": ["customer", "requesting", "refund", "damaged", "order"],
    },
    {
        "state": "Our production database crashed and we are losing critical data every minute.",
        "qid": "is_critical", "instructions": "Is this a critical emergency?",
        "key_words": ["production", "database", "crashed", "losing", "critical", "data"],
    },
    {
        "state": "The mobile application crashes every time the user tries to upload a photo.",
        "qid": "is_bug", "instructions": "Is this a bug report?",
        "key_words": ["mobile", "application", "crashes", "user", "upload", "photo"],
    },
    {
        "state": "I am extremely satisfied with your customer service and fast shipping.",
        "qid": "is_positive", "instructions": "Is the customer satisfied?",
        "key_words": ["extremely", "satisfied", "customer", "service", "fast", "shipping"],
    },
    {
        "state": "The competitor offers better pricing and more features for enterprise customers.",
        "qid": "is_competitor", "instructions": "Does this mention a competitor?",
        "key_words": ["competitor", "offers", "better", "pricing", "features", "enterprise"],
    },
    {
        "state": "Please cancel my subscription immediately and process a full refund.",
        "qid": "is_cancel", "instructions": "Is this a cancellation request?",
        "key_words": ["cancel", "subscription", "immediately", "process", "full", "refund"],
    },
    {
        "state": "The API endpoint returns a server error when sending large JSON payloads.",
        "qid": "is_technical", "instructions": "Is this a technical issue?",
        "key_words": ["API", "endpoint", "server", "error", "sending", "JSON", "payloads"],
    },
    {
        "state": "I would like to upgrade my plan from basic to premium for my team.",
        "qid": "is_upgrade", "instructions": "Is this an upgrade request?",
        "key_words": ["upgrade", "plan", "basic", "premium", "team"],
    },
    {
        "state": "My credit card was charged three times for the same purchase last Tuesday.",
        "qid": "is_billing", "instructions": "Is this a billing issue?",
        "key_words": ["credit", "card", "charged", "three", "times", "purchase"],
    },
    {
        "state": "The delivery arrived two weeks late and the package was completely destroyed.",
        "qid": "is_shipping", "instructions": "Is this a shipping problem?",
        "key_words": ["delivery", "arrived", "weeks", "late", "package", "destroyed"],
    },
]


def mask_words(text: str, words_to_mask: list[str], mask_token: str = "[REMOVED]") -> str:
    result = text
    for word in words_to_mask:
        result = result.replace(word, mask_token, 1)
    return result


# --- 3b-2: Bidirectional context placement ---
PADDING = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
)

BIDI_CASES = [
    {
        "hint": "Important context: the word 'bank' refers to a financial institution in this text.",
        "ambiguous": "The customer visited the bank to discuss their account and walked along the river afterward.",
        "qid": "meaning",
        "instructions": "Which type of bank is primarily discussed?",
        "criteria": {"financial": "A financial institution like a bank", "river": "The side of a river or waterway"},
        "expected": "financial",
    },
    {
        "hint": "Note: the customer is extremely frustrated and angry.",
        "ambiguous": "The customer wrote: I have a question about the status of my order. It has been a while since I placed it and I was wondering about the timeline.",
        "qid": "tone",
        "instructions": "Is the customer angry?",
        "criteria": None,  # noul
        "expected_high": True,
    },
    {
        "hint": "Context: this message is about a billing problem, not a technical issue.",
        "ambiguous": "Something went wrong with my account. I noticed some unexpected changes and I need help fixing it.",
        "qid": "dept",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "Payment and charges", "technical": "Software bugs and errors", "account": "Account settings and access"},
        "expected": "billing",
    },
    {
        "hint": "Background: the customer is a VIP enterprise client with a $500K annual contract.",
        "ambiguous": "I noticed a small issue on the settings page. When I click save, nothing happens.",
        "qid": "priority",
        "instructions": "Should this be treated as high priority?",
        "criteria": None,  # noul
        "expected_high": True,
    },
    {
        "hint": "Clarification: the customer has already been offered and rejected a replacement.",
        "ambiguous": "I want this resolved. The product I received does not match what was advertised.",
        "qid": "wants",
        "instructions": "Does the customer want a refund rather than a replacement?",
        "criteria": None,  # noul
        "expected_high": True,
    },
]


# --- 3b-3: Token-level noise ---
NOISE_WORDS = ["apple", "green", "table", "quickly", "seven", "blue", "random", "fish", "cloud", "stone"]

def insert_noise(text: str, noise_ratio: float) -> str:
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if random.random() < noise_ratio:
            result.append(random.choice(NOISE_WORDS))
    return " ".join(result)


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    print("=" * 70)
    print("PROBE 3b: Corrected Diffusion Signature Tests")
    print("=" * 70)

    # --- 3b-1: Word-level masking ---
    print(f"\n--- 3b-1: Word-level masking ({len(WORD_MASKING_CASES)} cases × 6 levels) ---")
    for i, case in enumerate(WORD_MASKING_CASES):
        key_words = case["key_words"]
        for mask_count in [0, 1, 2, 3, 4, min(5, len(key_words))]:
            words_to_mask = key_words[:mask_count]
            masked = mask_words(case["state"], words_to_mask) if mask_count > 0 else case["state"]
            t0 = time.monotonic()
            resp = client.noul(masked, case["qid"], case["instructions"])
            elapsed = time.monotonic() - t0
            emit({
                "test": "word_masking",
                "case_idx": i,
                "mask_count": mask_count,
                "total_key_words": len(key_words),
                "masked_words": words_to_mask,
                "state": masked,
                "noul": resp["answers"][case["qid"]]["noul"],
                "elapsed_s": round(elapsed, 4),
            })
        if (i + 1) % 3 == 0:
            print(f"  Processed {i + 1}/{len(WORD_MASKING_CASES)} cases")

    # --- 3b-2: Bidirectional context placement ---
    print(f"\n--- 3b-2: Bidirectional context ({len(BIDI_CASES)} cases × 3 placements) ---")
    padding = PADDING * 3  # ~600 chars of irrelevant padding
    for i, case in enumerate(BIDI_CASES):
        for placement in ["hint_first", "hint_last", "no_hint"]:
            if placement == "hint_first":
                state = case["hint"] + " " + padding + " " + case["ambiguous"]
            elif placement == "hint_last":
                state = case["ambiguous"] + " " + padding + " " + case["hint"]
            else:
                state = case["ambiguous"]

            t0 = time.monotonic()
            if case.get("criteria"):
                resp = client.choice(state, case["qid"], case["instructions"], case["criteria"])
                answer = resp["answers"][case["qid"]]
                result = {
                    "type": "choice",
                    "choice": answer["choice"],
                    "probabilities": answer["probabilities"],
                    "confidence": answer["confidence"],
                    "correct": answer["choice"] == case.get("expected"),
                }
            else:
                resp = client.noul(state, case["qid"], case["instructions"])
                answer = resp["answers"][case["qid"]]
                result = {
                    "type": "noul",
                    "noul": answer["noul"],
                }
            elapsed = time.monotonic() - t0

            emit({
                "test": "bidi_context",
                "case_idx": i,
                "placement": placement,
                "input_tokens": resp["usage"]["input_tokens"],
                "elapsed_s": round(elapsed, 4),
                **result,
            })
            print(f"  Case {i + 1} [{placement:10s}]: {result}")

    # --- 3b-3: Token-level noise ---
    print(f"\n--- 3b-3: Token-level noise ({len(WORD_MASKING_CASES)} cases × 5 noise levels) ---")
    for i, case in enumerate(WORD_MASKING_CASES):
        for noise_ratio in [0.0, 0.1, 0.2, 0.3, 0.5]:
            noisy = insert_noise(case["state"], noise_ratio) if noise_ratio > 0 else case["state"]
            t0 = time.monotonic()
            resp = client.noul(noisy, case["qid"], case["instructions"])
            elapsed = time.monotonic() - t0
            emit({
                "test": "token_noise",
                "case_idx": i,
                "noise_ratio": noise_ratio,
                "state": noisy,
                "noul": resp["answers"][case["qid"]]["noul"],
                "elapsed_s": round(elapsed, 4),
            })
        if (i + 1) % 3 == 0:
            print(f"  Processed {i + 1}/{len(WORD_MASKING_CASES)} cases")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    # Word masking
    wm = [r for r in records if r["test"] == "word_masking"]
    print("\nWord-level masking (mean noul by # words masked):")
    for mc in sorted(set(r["mask_count"] for r in wm)):
        vals = [r["noul"] for r in wm if r["mask_count"] == mc]
        print(f"  {mc} words masked: mean noul = {sum(vals)/len(vals):.4f} (n={len(vals)})")

    # Bidi context
    bidi = [r for r in records if r["test"] == "bidi_context"]
    print("\nBidirectional context placement:")
    for placement in ["no_hint", "hint_first", "hint_last"]:
        items = [r for r in bidi if r["placement"] == placement]
        noul_items = [r for r in items if r["type"] == "noul"]
        choice_items = [r for r in items if r["type"] == "choice"]
        if noul_items:
            mean_n = sum(r["noul"] for r in noul_items) / len(noul_items)
            print(f"  {placement:10s}: mean noul = {mean_n:.4f} (n={len(noul_items)})")
        if choice_items:
            correct = sum(1 for r in choice_items if r.get("correct"))
            print(f"  {placement:10s}: choice accuracy = {correct}/{len(choice_items)}")

    # Token noise
    tn = [r for r in records if r["test"] == "token_noise"]
    print("\nToken-level noise (mean noul by noise ratio):")
    for nr in sorted(set(r["noise_ratio"] for r in tn)):
        vals = [r["noul"] for r in tn if r["noise_ratio"] == nr]
        print(f"  {int(nr*100):3d}% noise: mean noul = {sum(vals)/len(vals):.4f} (n={len(vals)})")

    # Key comparison
    print(f"\n{'=' * 70}")
    print("KEY COMPARISON: hint_first vs hint_last")
    print("=" * 70)
    print("If hint_first >> hint_last → causal attention (early context dominates)")
    print("If hint_first ≈ hint_last → bidirectional attention (position doesn't matter)")
    for i, case in enumerate(BIDI_CASES):
        first_items = [r for r in bidi if r["case_idx"] == i and r["placement"] == "hint_first"]
        last_items = [r for r in bidi if r["case_idx"] == i and r["placement"] == "hint_last"]
        no_items = [r for r in bidi if r["case_idx"] == i and r["placement"] == "no_hint"]
        if first_items and last_items:
            f_val = first_items[0].get("noul") or first_items[0].get("probabilities", {}).get(case.get("expected"), "?")
            l_val = last_items[0].get("noul") or last_items[0].get("probabilities", {}).get(case.get("expected"), "?")
            n_val = no_items[0].get("noul") or no_items[0].get("probabilities", {}).get(case.get("expected"), "?")
            print(f"  Case {i}: no_hint={n_val}  hint_first={f_val}  hint_last={l_val}")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
