#!/usr/bin/env python3
"""Probe 3: Diffusion vs autoregressive signature.

Tests masking robustness, word scrambling, and cloze (fill-in-the-blank) tasks
to distinguish diffusion-based from autoregressive architectures.
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
OUTFILE = os.path.join(RESULTS_DIR, "03-diffusion-signature.jsonl")

random.seed(42)

# States with clear, unambiguous answers for masking tests
MASKING_STATES = [
    ("The customer is demanding a full refund for their order.", "is_refund", "Is the customer requesting a refund?", True),
    ("Everything arrived on time and in perfect condition. Thank you!", "is_happy", "Is the customer satisfied?", True),
    ("Our entire production database crashed and we are losing data.", "is_critical", "Is this a critical emergency?", True),
    ("Can you tell me what colors this shirt comes in?", "is_question", "Is this a product inquiry?", True),
    ("I will never buy from you again. This is the worst experience ever.", "is_angry", "Is the customer angry?", True),
    ("The API endpoint returns a 500 error on every POST request.", "is_technical", "Is this a technical bug report?", True),
    ("I would like to upgrade my subscription to the premium plan.", "is_upgrade", "Is this an upgrade request?", True),
    ("The package was delivered to the wrong address.", "is_shipping", "Is this a shipping problem?", True),
    ("My credit card was charged three times for the same order.", "is_billing", "Is this a billing issue?", True),
    ("The new dashboard feature is really intuitive and saves me time.", "is_praise", "Is this positive feedback?", True),
    ("How do I reset my password?", "is_password", "Is this about account access?", True),
    ("I noticed the price changed after I added it to my cart.", "is_pricing", "Is this about pricing?", True),
    ("Your mobile app keeps crashing on Android 14.", "is_bug", "Is this a bug report?", True),
    ("I want to cancel my subscription effective immediately.", "is_cancel", "Is this a cancellation request?", True),
    ("The documentation says the endpoint accepts JSON but it only works with form data.", "is_doc_issue", "Is this about incorrect documentation?", True),
    ("Just checking in to see if my ticket has been assigned to someone.", "is_followup", "Is this a follow-up on an existing ticket?", True),
    ("We need to process payroll by Friday and the export feature is broken.", "is_deadline", "Is there a deadline mentioned?", True),
    ("Can I get a sample before placing a bulk order?", "is_sample", "Is this a sample request?", True),
    ("The search function returns no results even for common terms.", "is_search_bug", "Is this about broken search functionality?", True),
    ("Thank you for the quick resolution! I appreciate the help.", "is_thanks", "Is this a thank-you message?", True),
]


def mask_text(text: str, mask_pct: float) -> str:
    chars = list(text)
    num_to_mask = int(len(chars) * mask_pct)
    indices = [i for i, c in enumerate(chars) if c.isalpha()]
    to_mask = random.sample(indices, min(num_to_mask, len(indices)))
    for idx in to_mask:
        chars[idx] = "_"
    return "".join(chars)


def scramble_words(text: str) -> str:
    words = text.split()
    random.shuffle(words)
    return " ".join(words)


def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])


CLOZE_TESTS = [
    ("The customer ordered a [MASK] but received the wrong size.", "product",
     {"shoes": "Footwear", "laptop": "Computer", "book": "Reading material", "phone": "Mobile device"}, "shoes"),
    ("The payment was processed in [MASK] and the customer wants USD.", "currency",
     {"euros": "European currency", "yen": "Japanese currency", "pounds": "British currency", "bitcoin": "Cryptocurrency"}, "euros"),
    ("The server is running [MASK] and needs to be upgraded.", "os",
     {"linux": "Open source OS", "windows": "Microsoft OS", "macos": "Apple OS", "freebsd": "BSD variant"}, "linux"),
    ("The customer's primary language is [MASK].", "language",
     {"english": "English language", "spanish": "Spanish language", "chinese": "Chinese language", "french": "French language"}, "english"),
    ("The bug only occurs in the [MASK] browser.", "browser",
     {"chrome": "Google Chrome", "safari": "Apple Safari", "firefox": "Mozilla Firefox", "edge": "Microsoft Edge"}, "chrome"),
    ("The data center is located in [MASK].", "location",
     {"virginia": "US East Coast", "oregon": "US West Coast", "frankfurt": "Europe", "tokyo": "Asia Pacific"}, "virginia"),
    ("The application is built with [MASK].", "framework",
     {"react": "Facebook UI library", "django": "Python web framework", "express": "Node.js framework", "rails": "Ruby framework"}, "react"),
    ("The file format is [MASK].", "format",
     {"json": "JavaScript Object Notation", "csv": "Comma-separated values", "xml": "Extensible Markup Language", "yaml": "YAML Ain't Markup Language"}, "json"),
]


def main():
    client = JevClient()
    records = []

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Clear output file
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 3: Diffusion vs Autoregressive Signature")
    print("=" * 70)

    # --- 3a: Masking robustness ---
    print(f"\n--- 3a: Masking robustness ({len(MASKING_STATES)} states × 6 masking levels) ---")
    for i, (state, qid, instruction, _) in enumerate(MASKING_STATES):
        for mask_pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            masked = mask_text(state, mask_pct) if mask_pct > 0 else state
            t0 = time.monotonic()
            resp = client.noul(masked, qid, instruction)
            elapsed = time.monotonic() - t0
            noul_val = resp["answers"][qid]["noul"]
            emit({
                "test": "masking",
                "state_idx": i,
                "mask_pct": mask_pct,
                "original": state,
                "masked": masked,
                "noul": noul_val,
                "elapsed_s": round(elapsed, 4),
            })
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(MASKING_STATES)} states")

    # --- 3b: Word scrambling ---
    print(f"\n--- 3b: Word scrambling ({len(MASKING_STATES)} states × 3 orderings) ---")
    for i, (state, qid, instruction, _) in enumerate(MASKING_STATES):
        for order_name, transform in [("original", lambda s: s), ("reversed", reverse_words), ("shuffled", scramble_words)]:
            transformed = transform(state)
            t0 = time.monotonic()
            resp = client.noul(transformed, qid, instruction)
            elapsed = time.monotonic() - t0
            noul_val = resp["answers"][qid]["noul"]
            emit({
                "test": "scrambling",
                "state_idx": i,
                "order": order_name,
                "original": state,
                "transformed": transformed,
                "noul": noul_val,
                "elapsed_s": round(elapsed, 4),
            })
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(MASKING_STATES)} states")

    # --- 3c: Cloze (fill-in-the-blank) ---
    print(f"\n--- 3c: Cloze tests ({len(CLOZE_TESTS)} items) ---")
    for i, (state, qid, options, expected) in enumerate(CLOZE_TESTS):
        t0 = time.monotonic()
        resp = client.choice(state, qid, f"What word best fills the [MASK] in the text?", options)
        elapsed = time.monotonic() - t0
        answer = resp["answers"][qid]
        emit({
            "test": "cloze",
            "state_idx": i,
            "state": state,
            "expected": expected,
            "choice": answer["choice"],
            "correct": answer["choice"] == expected,
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "elapsed_s": round(elapsed, 4),
        })
        print(f"  Cloze {i + 1}: expected={expected}, got={answer['choice']} "
              f"(p={answer['probabilities'].get(expected, 0):.3f}, conf={answer['confidence']:.3f})")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    mask_records = [r for r in records if r["test"] == "masking"]
    print(f"\nMasking ({len(mask_records)} records):")
    for pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        nouls = [r["noul"] for r in mask_records if r["mask_pct"] == pct]
        mean_noul = sum(nouls) / len(nouls) if nouls else 0
        print(f"  {int(pct * 100):3d}% masked: mean noul = {mean_noul:.4f} (n={len(nouls)})")

    scramble_records = [r for r in records if r["test"] == "scrambling"]
    print(f"\nScrambling ({len(scramble_records)} records):")
    for order in ["original", "reversed", "shuffled"]:
        nouls = [r["noul"] for r in scramble_records if r["order"] == order]
        mean_noul = sum(nouls) / len(nouls) if nouls else 0
        print(f"  {order:10s}: mean noul = {mean_noul:.4f} (n={len(nouls)})")

    cloze_records = [r for r in records if r["test"] == "cloze"]
    correct = sum(1 for r in cloze_records if r["correct"])
    print(f"\nCloze ({len(cloze_records)} items): {correct}/{len(cloze_records)} correct")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
