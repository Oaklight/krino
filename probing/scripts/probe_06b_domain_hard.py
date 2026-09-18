#!/usr/bin/env python3
"""Probe 6b: Domain variance with HARD questions near capability boundary.

Probe 6 flaw: 100% ceiling across all domains. Questions were too easy.
This probe uses tricky, counterintuitive, or expert-level questions
where errors are expected. We need to see WHERE the model fails.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "06b-domain-hard.jsonl")

# Intentionally tricky questions — many are counterintuitive
DOMAINS = {
    "general_tricky": [
        ("Glass is a liquid that flows very slowly over time.", "q", "Is this a correct description of glass?", False),
        ("Irregardless is not a real English word.", "q", "Is this claim correct?", False),  # it IS in dictionaries
        ("Napoleon Bonaparte was unusually short for his time.", "q", "Is this historically accurate?", False),
        ("Humans use only 10% of their brains.", "q", "Is this scientifically accurate?", False),
        ("The tongue has distinct regions for different tastes.", "q", "Is the tongue map theory correct?", False),
        ("Chameleons change color primarily for camouflage.", "q", "Is this the primary reason?", False),
        ("Spinach is an exceptionally good source of iron compared to other vegetables.", "q", "Is this correct?", False),
        ("Bats are completely blind.", "q", "Is this correct?", False),
        ("Fortune cookies originated in China.", "q", "Is this historically correct?", False),
        ("Sugar causes hyperactivity in children.", "q", "Is this scientifically supported?", False),
    ],
    "code_tricky": [
        ("In Python 3, 0.1 + 0.2 == 0.3 evaluates to True.", "q", "Is this correct?", False),
        ("In JavaScript, typeof null returns 'null'.", "q", "Is this correct?", False),  # returns 'object'
        ("In C, the size of an int is always 4 bytes.", "q", "Is this guaranteed by the C standard?", False),
        ("In Python, a tuple with one element is written as (1).", "q", "Is this correct syntax for a tuple?", False),  # needs trailing comma
        ("In Git, a fast-forward merge always creates a merge commit.", "q", "Is this correct?", False),
        ("In SQL, COUNT(*) and COUNT(column) always return the same result.", "q", "Is this correct?", False),  # COUNT(col) skips NULLs
        ("In Python, 'is' and '==' are interchangeable for comparing integers.", "q", "Is this always correct?", False),
        ("In JavaScript, [] == false evaluates to true.", "q", "Is this correct?", True),  # JS type coercion
        ("In Rust, all variables are mutable by default.", "q", "Is this correct?", False),
        ("A HashMap in Java guarantees insertion order.", "q", "Is this correct?", False),  # LinkedHashMap does
    ],
    "medical_tricky": [
        ("Cracking your knuckles causes arthritis.", "q", "Is this medically supported?", False),
        ("You should tilt your head back during a nosebleed.", "q", "Is this the recommended first aid?", False),
        ("Eating before swimming causes dangerous cramps.", "q", "Is this medically accurate?", False),
        ("Reading in dim light permanently damages your eyes.", "q", "Is this correct?", False),
        ("You need to drink exactly 8 glasses of water per day.", "q", "Is this a medically established requirement?", False),
        ("Vaccines can cause autism.", "q", "Is this scientifically supported?", False),
        ("Cold weather directly causes the common cold.", "q", "Is this the correct cause?", False),
        ("You lose most body heat through your head.", "q", "Is this disproportionately true?", False),
        ("Organic food is more nutritious than conventional food.", "q", "Is this consistently supported by evidence?", False),
        ("Blood in veins is blue.", "q", "Is this correct?", False),
    ],
    "science_tricky": [
        ("The North Star (Polaris) is the brightest star in the night sky.", "q", "Is this correct?", False),  # Sirius is
        ("The Great Red Spot on Jupiter is a solid surface feature.", "q", "Is this correct?", False),  # it's a storm
        ("A full moon can affect human behavior (the lunar effect).", "q", "Is this scientifically established?", False),
        ("Evolution means species always become more complex over time.", "q", "Is this a correct description of evolution?", False),
        ("The seasons are caused by Earth's distance from the Sun.", "q", "Is this the correct cause?", False),  # axial tilt
        ("Lightning never strikes the same place twice.", "q", "Is this physically correct?", False),
        ("Dropping a penny from the Empire State Building could kill someone.", "q", "Is this physically plausible?", False),
        ("Water always drains clockwise in the Northern Hemisphere due to the Coriolis effect.", "q", "Is this correct for household drains?", False),
        ("Diamonds are the hardest material known to science.", "q", "Is this still correct?", False),  # wurtzite boron nitride
        ("We can see the Great Wall of China from the Moon.", "q", "Is this correct?", False),
    ],
    "finance_tricky": [
        ("A stock split increases the total value of a shareholder's position.", "q", "Is this correct?", False),
        ("Government bonds are completely risk-free investments.", "q", "Is this correct?", False),
        ("The GDP of a country always increases when its stock market goes up.", "q", "Is this a reliable relationship?", False),
        ("Dollar cost averaging always outperforms lump sum investing.", "q", "Is this correct?", False),
        ("A company with higher revenue always has higher profits.", "q", "Is this necessarily true?", False),
        ("Deflation is always bad for an economy.", "q", "Is this universally true?", False),
        ("Gold is a reliable hedge against inflation in all time periods.", "q", "Is this consistently true?", False),
        ("Mutual funds always outperform index funds.", "q", "Is this correct?", False),
        ("A country's trade deficit is always an indicator of economic weakness.", "q", "Is this necessarily true?", False),
        ("Raising the minimum wage always increases unemployment.", "q", "Is this universally supported by evidence?", False),
    ],
    "legal_tricky": [
        ("In the US, it is illegal to record a phone call without both parties' consent.", "q", "Is this correct in all states?", False),
        ("A verbal contract is never legally binding.", "q", "Is this correct?", False),
        ("You have the right to a phone call when arrested in the US.", "q", "Is this a constitutional right?", False),
        ("Entrapment means the police can never use undercover operations.", "q", "Is this the correct legal definition?", False),
        ("You can be arrested for refusing to show ID to police in the US.", "q", "Is this true in all states?", False),
        ("Copyright must be registered to exist.", "q", "Is this correct under current law?", False),
        ("Self-defense always justifies the use of lethal force.", "q", "Is this universally true?", False),
        ("A landlord can enter a rental property at any time.", "q", "Is this generally correct?", False),
        ("Freedom of speech means you cannot be fired for what you say.", "q", "Is this correct?", False),
        ("All crimes require criminal intent (mens rea).", "q", "Is this correct?", False),  # strict liability
    ],
}


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 6b: Hard Domain Variance")
    print(f"Domains: {len(DOMAINS)}, Questions per domain: 10")
    print("NOTE: Most answers should be FALSE (counterintuitive questions)")
    print("=" * 70)

    for domain, questions in DOMAINS.items():
        correct = 0
        total = 0
        for state, qid, instructions, expected in questions:
            t0 = time.monotonic()
            resp = client.noul(state, qid, instructions)
            elapsed = time.monotonic() - t0
            noul = resp["answers"][qid]["noul"]
            predicted = noul > 0.5
            is_correct = predicted == expected

            record = {
                "domain": domain,
                "state": state,
                "noul": noul,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "elapsed_s": round(elapsed, 4),
            }
            records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            total += 1
            if is_correct:
                correct += 1

        print(f"  {domain:20s}: {correct}/{total} ({100 * correct / total:.0f}%)")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    domain_accs = {}
    for domain in DOMAINS:
        dr = [r for r in records if r["domain"] == domain]
        acc = sum(1 for r in dr if r["correct"]) / len(dr)
        domain_accs[domain] = acc

    mean_acc = sum(domain_accs.values()) / len(domain_accs)
    variance = sum((a - mean_acc) ** 2 for a in domain_accs.values()) / len(domain_accs)
    std_dev = variance ** 0.5

    print("\nRanked by accuracy:")
    for domain, acc in sorted(domain_accs.items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 30)
        print(f"  {domain:20s}: {acc:.0%}  {bar}")

    print(f"\nMean: {mean_acc:.0%}, Std: {std_dev:.4f}, Range: {min(domain_accs.values()):.0%}-{max(domain_accs.values()):.0%}")
    print(f"MoE signal: {'HIGH' if std_dev > 0.10 else 'MODERATE' if std_dev > 0.05 else 'LOW'} (σ={std_dev:.4f})")

    # List wrong answers
    wrong = [r for r in records if not r["correct"]]
    if wrong:
        print(f"\nIncorrect answers ({len(wrong)}):")
        for r in wrong:
            print(f"  [{r['domain']}] noul={r['noul']:.2f} expected={r['expected']} | {r['state'][:70]}")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
