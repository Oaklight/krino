#!/usr/bin/env python3
"""Probe 6: MoE vs dense — domain capability variance.

Tests whether Jev shows uneven domain capabilities consistent with sparse MoE
routing vs uniform dense model behavior.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "06-domain-variance.jsonl")

# 15 questions per domain, each with a known correct answer
DOMAINS = {
    "general_knowledge": [
        ("The capital of Australia is not Sydney.", "capital", "Is the statement correct?", True),
        ("Water boils at 100 degrees Celsius at sea level.", "boiling", "Is this scientifically accurate?", True),
        ("The Great Wall of China is visible from space with the naked eye.", "wall", "Is this claim true?", False),
        ("Lightning never strikes the same place twice.", "lightning", "Is this a true statement?", False),
        ("Humans share approximately 98% of their DNA with chimpanzees.", "dna", "Is this approximately correct?", True),
        ("The Sahara is the largest desert on Earth.", "sahara", "Is this correct? Consider all desert types.", False),
        ("Goldfish have a memory span of only three seconds.", "goldfish", "Is this claim scientifically supported?", False),
        ("Mount Everest is the tallest mountain measured from sea level.", "everest", "Is this correct?", True),
        ("Bananas are technically berries.", "banana", "Is this botanically correct?", True),
        ("The human body has 206 bones in adulthood.", "bones", "Is this the correct number?", True),
        ("Diamonds are made from compressed coal.", "diamonds", "Is this the correct formation process?", False),
        ("Octopuses have three hearts.", "octopus", "Is this anatomically correct?", True),
        ("The Amazon River is the longest river in the world.", "amazon", "Is this correct?", False),
        ("Sound travels faster in water than in air.", "sound", "Is this physically correct?", True),
        ("Vikings wore horned helmets.", "vikings", "Is this historically accurate?", False),
    ],
    "code_programming": [
        ("In Python, 'is' checks identity while '==' checks equality.", "python_is", "Is this description correct?", True),
        ("In JavaScript, '===' performs type coercion before comparison.", "js_strict", "Is this correct?", False),
        ("A binary search requires the input array to be sorted.", "bsearch", "Is this a correct prerequisite?", True),
        ("In Git, 'git rebase' preserves the original commit hashes.", "git_rebase", "Is this correct?", False),
        ("TCP guarantees in-order delivery of packets.", "tcp", "Is this correct?", True),
        ("In SQL, NULL = NULL evaluates to TRUE.", "sql_null", "Is this correct?", False),
        ("A deadlock requires at least two threads.", "deadlock", "Is this a necessary condition?", True),
        ("In Python, lists are immutable.", "py_list", "Is this correct?", False),
        ("HTTP is a stateful protocol.", "http", "Is this correct?", False),
        ("In Big-O notation, O(n log n) is faster than O(n^2) for large n.", "bigO", "Is this correct?", True),
        ("Rust's borrow checker runs at runtime.", "rust_borrow", "Is this correct?", False),
        ("A stack uses FIFO ordering.", "stack", "Is this correct?", False),
        ("DNS uses UDP by default for queries.", "dns", "Is this correct?", True),
        ("In Python, a generator yields values lazily.", "py_gen", "Is this correct?", True),
        ("Docker containers share the host OS kernel.", "docker", "Is this correct?", True),
    ],
    "medical_clinical": [
        ("Aspirin is an anticoagulant medication.", "aspirin", "Is this the correct drug classification?", False),
        ("Type 1 diabetes is an autoimmune condition.", "t1d", "Is this the correct pathophysiology?", True),
        ("Normal resting heart rate for adults is 60-100 beats per minute.", "hr", "Is this the correct range?", True),
        ("Antibiotics are effective against viral infections.", "antibiotics", "Is this correct?", False),
        ("The liver is the largest internal organ in the human body.", "liver", "Is this correct?", True),
        ("Hypertension is defined as blood pressure above 140/90 mmHg.", "bp", "Is this the standard threshold?", True),
        ("Insulin is produced by the liver.", "insulin", "Is this the correct organ?", False),
        ("A CT scan uses magnetic fields to produce images.", "ct", "Is this the correct imaging mechanism?", False),
        ("Penicillin was discovered by Alexander Fleming.", "penicillin", "Is this historically correct?", True),
        ("The normal body temperature is exactly 98.6°F with no variation.", "temp", "Is this precisely correct?", False),
        ("Anemia is a condition characterized by low red blood cell count.", "anemia", "Is this a correct description?", True),
        ("The appendix has no known function in the human body.", "appendix", "Is this the current medical consensus?", False),
        ("MRSA is resistant to methicillin.", "mrsa", "Is this correct by definition?", True),
        ("A normal fasting blood glucose is under 100 mg/dL.", "glucose", "Is this the correct threshold?", True),
        ("An ECG measures brain electrical activity.", "ecg", "Is this the correct function?", False),
    ],
    "legal": [
        ("In US law, you are guilty until proven innocent.", "presumption", "Is this the correct legal standard?", False),
        ("A tort is a civil wrong.", "tort", "Is this the correct legal definition?", True),
        ("The First Amendment protects freedom of speech from government restriction.", "1a", "Is this correct?", True),
        ("A contract requires consideration from both parties.", "consideration", "Is this a correct requirement?", True),
        ("Double jeopardy means you can be tried twice for the same crime.", "dj", "Is this the correct meaning?", False),
        ("Habeas corpus is a writ to produce a prisoner before the court.", "habeas", "Is this correct?", True),
        ("Copyright protection lasts for 25 years.", "copyright", "Is this the correct duration in the US?", False),
        ("An NDA is a Non-Disclosure Agreement.", "nda", "Is this the correct expansion?", True),
        ("In common law, precedent from higher courts is binding.", "precedent", "Is this correct?", True),
        ("A felony is always more serious than a misdemeanor.", "felony", "Is this generally correct in US law?", True),
        ("Miranda rights must be read before any police interaction.", "miranda", "Is this correct?", False),
        ("Res judicata prevents the same case from being relitigated.", "res_jud", "Is this the correct principle?", True),
        ("In patent law, an invention must be novel to be patentable.", "patent", "Is this a correct requirement?", True),
        ("Eminent domain allows the government to take private property without compensation.", "eminent", "Is this fully correct?", False),
        ("Statutory law is created by courts through judicial decisions.", "statutory", "Is this the correct source?", False),
    ],
    "finance": [
        ("A bond's price moves inversely to interest rates.", "bond", "Is this the correct relationship?", True),
        ("The P/E ratio divides stock price by earnings per share.", "pe", "Is this the correct formula?", True),
        ("Diversification eliminates all investment risk.", "diversification", "Is this correct?", False),
        ("EBITDA stands for Earnings Before Interest, Taxes, Depreciation and Amortization.", "ebitda", "Is this the correct expansion?", True),
        ("Short selling profits when a stock price goes up.", "short", "Is this the correct direction?", False),
        ("The Federal Reserve sets fiscal policy.", "fed", "Is this the correct role?", False),
        ("Compound interest earns interest on previously earned interest.", "compound", "Is this the correct definition?", True),
        ("A bull market is characterized by falling stock prices.", "bull", "Is this correct?", False),
        ("Liquidity refers to how easily an asset can be converted to cash.", "liquidity", "Is this the correct definition?", True),
        ("GDP stands for Gross Domestic Product.", "gdp", "Is this the correct expansion?", True),
        ("Inflation always decreases the purchasing power of money.", "inflation", "Is this generally correct?", True),
        ("An IPO is an Initial Private Offering.", "ipo", "Is this the correct expansion?", False),
        ("Callable bonds can be redeemed by the issuer before maturity.", "callable", "Is this correct?", True),
        ("The yield curve normally slopes upward.", "yield", "Is this correct for normal conditions?", True),
        ("A hedge fund is available to all retail investors.", "hedge", "Is this generally correct?", False),
    ],
    "science_physics": [
        ("The speed of light in a vacuum is approximately 300,000 km/s.", "light", "Is this approximately correct?", True),
        ("Electrons are larger than protons.", "electron", "Is this correct?", False),
        ("Entropy in a closed system always decreases over time.", "entropy", "Is this correct?", False),
        ("Newton's third law states every action has an equal and opposite reaction.", "newton3", "Is this correct?", True),
        ("Absolute zero is 0 degrees Celsius.", "abs_zero", "Is this correct?", False),
        ("The Higgs boson gives particles their mass.", "higgs", "Is this a simplified but correct description?", True),
        ("Black holes emit no radiation whatsoever.", "bh_rad", "Is this correct?", False),
        ("Photons have no mass.", "photon", "Is this correct?", True),
        ("The strong nuclear force is the weakest fundamental force.", "strong", "Is this correct?", False),
        ("Special relativity says nothing can travel faster than light.", "relativity", "Is this correct?", True),
        ("Superconductivity occurs at extremely high temperatures.", "supercon", "Is this correct for conventional superconductors?", False),
        ("Quarks come in six flavors.", "quarks", "Is this the correct number?", True),
        ("Sound can travel through a vacuum.", "sound_vac", "Is this correct?", False),
        ("E=mc² relates energy and mass.", "emc2", "Is this correct?", True),
        ("Plasma is the fourth state of matter.", "plasma", "Is this correct?", True),
    ],
    "multilingual": [
        ("'Bonjour' means 'goodbye' in French.", "bonjour", "Is this translation correct?", False),
        ("Japanese uses three writing systems: hiragana, katakana, and kanji.", "jp_writing", "Is this correct?", True),
        ("'Danke' means 'thank you' in German.", "danke", "Is this correct?", True),
        ("Mandarin Chinese is a tonal language with four tones.", "mandarin", "Is this correct?", True),
        ("Spanish and Portuguese are mutually intelligible.", "es_pt", "Is this generally true?", True),
        ("Arabic is written from left to right.", "arabic_dir", "Is this correct?", False),
        ("Korean uses an alphabetic writing system called Hangul.", "hangul", "Is this correct?", True),
        ("'Gracias' means 'please' in Spanish.", "gracias", "Is this correct?", False),
        ("Russian uses the Cyrillic alphabet.", "russian", "Is this correct?", True),
        ("Hindi and Urdu are linguistically the same language.", "hindi_urdu", "Is this roughly correct?", True),
        ("'Sayonara' is a casual way to say goodbye in Japanese.", "sayonara", "Is this correct?", False),
        ("There are approximately 7,000 languages spoken in the world today.", "languages", "Is this approximately correct?", True),
        ("Thai is a tonal language.", "thai", "Is this correct?", True),
        ("Latin is still an official language of Vatican City.", "latin", "Is this correct?", True),
        ("Finnish belongs to the Germanic language family.", "finnish", "Is this correct?", False),
    ],
    "pop_culture": [
        ("The Beatles were from Liverpool, England.", "beatles", "Is this correct?", True),
        ("Star Wars was directed by Steven Spielberg.", "star_wars", "Is this correct for the original film?", False),
        ("The Marvel Cinematic Universe began with Iron Man in 2008.", "mcu", "Is this correct?", True),
        ("Harry Potter attended Hogwarts School of Witchcraft and Wizardry.", "hp", "Is this correct?", True),
        ("The Lord of the Rings was written by C.S. Lewis.", "lotr", "Is this correct?", False),
        ("Michael Jordan played for the Chicago Bulls.", "mj", "Is this correct?", True),
        ("The Super Bowl is the championship game of the NBA.", "superbowl", "Is this correct?", False),
        ("Minecraft was originally created by Markus Persson.", "minecraft", "Is this correct?", True),
        ("The Simpsons first aired in the 2000s.", "simpsons", "Is this correct?", False),
        ("Instagram was acquired by Facebook.", "instagram", "Is this correct?", True),
        ("Spotify is primarily a video streaming service.", "spotify", "Is this correct?", False),
        ("The Olympics are held every 3 years.", "olympics", "Is this the correct frequency?", False),
        ("Beyoncé was a member of Destiny's Child.", "beyonce", "Is this correct?", True),
        ("Tesla was founded by Elon Musk.", "tesla_founded", "Is Elon Musk the original founder?", False),
        ("Netflix started as a DVD rental service.", "netflix", "Is this historically correct?", True),
    ],
}


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 6: MoE vs Dense — Domain Capability Variance")
    print(f"Domains: {len(DOMAINS)}, Questions per domain: 15")
    print("=" * 70)

    for domain, questions in DOMAINS.items():
        correct = 0
        total = 0
        for state, qid, instructions, expected_true in questions:
            t0 = time.monotonic()
            resp = client.noul(state, qid, instructions)
            elapsed = time.monotonic() - t0
            noul = resp["answers"][qid]["noul"]
            predicted_true = noul > 0.5
            is_correct = predicted_true == expected_true

            record = {
                "domain": domain,
                "state": state,
                "qid": qid,
                "noul": noul,
                "expected": expected_true,
                "predicted": predicted_true,
                "correct": is_correct,
                "elapsed_s": round(elapsed, 4),
            }
            records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            total += 1
            if is_correct:
                correct += 1

        acc = correct / total
        print(f"  {domain:25s}: {correct}/{total} correct ({acc:.1%})")

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

    print(f"\nPer-domain accuracy:")
    for domain, acc in sorted(domain_accs.items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {domain:25s}: {acc:.1%}  {bar}")

    print(f"\nMean accuracy: {mean_acc:.1%}")
    print(f"Std deviation: {std_dev:.4f}")
    print(f"Min: {min(domain_accs.values()):.1%} ({min(domain_accs, key=domain_accs.get)})")
    print(f"Max: {max(domain_accs.values()):.1%} ({max(domain_accs, key=domain_accs.get)})")
    print(f"Range: {max(domain_accs.values()) - min(domain_accs.values()):.1%}")

    print(f"\nMoE signal: {'HIGH' if std_dev > 0.10 else 'LOW'} variance (σ={std_dev:.4f})")
    print(f"Total: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
