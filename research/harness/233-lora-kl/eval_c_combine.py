"""Combine three per-seed standard C gates under the pre-registered rule.

No pooled test.  All three seeds score the same 6,254 held-out items, so their
per-item results are correlated; concatenating them would treat 6,254
correlated observations as 18,762 independent ones and inflate significance by
roughly sqrt(3).  Promotion needs, from canon_preregistered_gate_rule and its
amendment:

- unanimous JevBench admissibility (all three seeds);
- a positive held-out delta in every seed;
- exact McNemar significance on the held-out in at least two of three.

Two of three significant with the third positive promotes.  Two of three
significant with the third negative does not promote and is reported unstable.
Three positive with one significant does not promote.
"""

import argparse
import json
import os


def atomic_dump(payload, path):
    """Write JSON so a reader never sees a partial or stale file.

    Removing the final path first means a crash before the rename leaves no
    verdict at all, rather than a previous run's valid-looking one.
    """
    if os.path.exists(path):
        os.remove(path)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def combine(gates):
    if len(gates) != 3:
        raise ValueError(f"the rule is defined over three seeds, got {len(gates)}")
    records = []
    for gate in gates:
        jev = gate["jevbench"]
        held = gate["heldout"]
        records.append(
            {
                "seed_dir": gate.get("seed_dir"),
                "admissible": bool(jev["admissible"]),
                "jevbench_cc": jev["metrics"]["cc"],
                "jevbench_paired": jev["paired_vs_control"],
                "heldout_n": held["metrics"]["n"],
                "heldout_cc": held["metrics"]["cc"],
                "heldout_base_cc": held["base_metrics"]["cc"],
                "heldout_delta": held["delta_cc"],
                "heldout_paired": held["paired_vs_base"],
                "significant": bool(held["significant"]),
            }
        )
    admissible = [record["admissible"] for record in records]
    deltas = [record["heldout_delta"] for record in records]
    significant = [record["significant"] for record in records]
    n_values = {record["heldout_n"] for record in records}

    unanimous_admissible = all(admissible)
    positive_all = all(delta > 0 for delta in deltas)
    significant_count = sum(significant)
    promoted = unanimous_admissible and positive_all and significant_count >= 2

    if promoted:
        status = "promoted"
    elif not unanimous_admissible:
        status = "not_promoted_admissibility"
    elif significant_count >= 2 and not positive_all:
        status = "not_promoted_unstable"
    elif not positive_all:
        status = "not_promoted_nonpositive_seed"
    else:
        status = "not_promoted_insufficient_significance"

    return {
        "rule": "canon_preregistered_gate_rule + amendment",
        "seeds": records,
        "admissible_unanimous": unanimous_admissible,
        "admissible_failing_seeds": [
            record["seed_dir"] for record in records if not record["admissible"]
        ],
        "heldout_deltas": deltas,
        "heldout_delta_positive_all": positive_all,
        "heldout_delta_negative_seeds": [
            record["seed_dir"] for record in records if record["heldout_delta"] < 0
        ],
        "heldout_delta_zero_seeds": [
            record["seed_dir"] for record in records if record["heldout_delta"] == 0
        ],
        "heldout_significant_count": significant_count,
        "heldout_delta_spread": max(deltas) - min(deltas),
        "heldout_delta_mean": sum(deltas) / len(deltas),
        "heldout_n_values": sorted(n_values),
        "pooled_test": "none: seeds share the same items; a pooled test needs a "
        "seed-as-random-effect treatment, not concatenation",
        "status": status,
        "promoted": promoted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gates", nargs=3, required=True, help="three per-seed gate_c.json files"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gates = [json.load(open(path)) for path in args.gates]
    result = combine(gates)
    atomic_dump(result, args.output)
    print(json.dumps(result, indent=1))
    deltas = " ".join(f"{record['heldout_delta']:+.4f}" for record in result["seeds"])
    flags = " ".join(
        "sig" if record["significant"] else "n.s." for record in result["seeds"]
    )
    admissible = " ".join(
        "ok" if record["admissible"] else "FAIL" for record in result["seeds"]
    )
    print(
        f"seed deltas {deltas}; significance {flags}; admissibility {admissible}; "
        f"status {result['status']}"
    )


if __name__ == "__main__":
    main()
