"""Standard C, dual gate: JevBench hard constraint and frozen held-out signal.

Both numbers are reported side by side.  JevBench admissible requires cc not
below the control 0.5187 and no significant McNemar loss.  Promotion requires
JevBench admissible AND a significant paired improvement on the frozen
held-out.  Neither alone promotes.
"""

import argparse
import json
import math
import os
from collections import Counter

from data.pipeline import load_jevbench


def atomic_dump(payload, path):
    """Write JSON so a reader never sees a partial or stale file.

    The final path is removed first: if the run crashes before the rename, it
    leaves neither a fresh verdict nor a previous run's stale one, and the
    combine's existence guard then sees no gate file at all.
    """
    if os.path.exists(path):
        os.remove(path)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def corrected(row):
    k = row["n_options"]
    return (float(bool(row["correct"])) - 1.0 / k) / (1.0 - 1.0 / k)


def ece(rows, n_bins=10):
    if not rows:
        return 0.0
    confidences = [max(row["probs"]) for row in rows]
    correct = [bool(row["correct"]) for row in rows]
    total = len(rows)
    error = 0.0
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        members = [
            i
            for i, confidence in enumerate(confidences)
            if (low <= confidence < high) or (index == n_bins - 1 and confidence == high)
        ]
        if not members:
            continue
        avg_conf = sum(confidences[i] for i in members) / len(members)
        avg_acc = sum(1.0 if correct[i] else 0.0 for i in members) / len(members)
        error += abs(avg_acc - avg_conf) * len(members) / total
    return error


def summarize(rows):
    usable = [row for row in rows if row.get("n_options", 0) > 1]
    if not usable:
        return {"n": 0}
    values = [corrected(row) for row in usable]
    per_type = {}
    for q_type in ("noul", "choice", "score"):
        subset = [row for row in usable if row.get("q_type") == q_type]
        if subset:
            per_type[q_type] = {
                "n": len(subset),
                "cc": sum(corrected(row) for row in subset) / len(subset),
            }
    return {
        "n": len(usable),
        "accuracy": sum(bool(row["correct"]) for row in usable) / len(usable),
        "cc": sum(values) / len(values),
        "ece": ece(usable),
        "by_type": per_type,
    }


def label_index(question, label):
    q_type = question["type"]
    if q_type == "noul":
        return 1 if bool(label) else 0
    if q_type == "choice":
        keys = list(question.get("criteria", {}))
        return keys.index(label) if label in keys else -1
    return int(round(float(label)))


def fractional_score_rows(items):
    """Score rows whose label is not an integer, and the score-row total."""
    score = [item for item in items if item["question"]["type"] == "score"]
    fractional = [
        item for item in score
        if float(item["label"]) != int(round(float(item["label"])))
    ]
    return len(fractional), len(score)


def load_heldout_base(items_path, base_probs_path, heldout_ids):
    items = {}
    with open(items_path) as handle:
        for line in handle:
            record = json.loads(line)
            items[record["id"]] = record
    base = {}
    with open(base_probs_path) as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] not in heldout_ids:
                continue
            item = items[record["id"]]
            target = label_index(item["question"], item["label"])
            probs = record["probs"]
            if target < 0 or not probs:
                continue
            prediction = max(range(len(probs)), key=lambda i: probs[i])
            base[record["id"]] = {
                "id": record["id"],
                "q_type": item["question"]["type"],
                "n_options": len(probs),
                "correct": prediction == target,
                "probs": probs,
            }
    return base


def paired(arm_rows, control_rows):
    control = {row["id"]: bool(row["correct"]) for row in control_rows}
    wins = losses = 0
    for row in arm_rows:
        if row["id"] not in control:
            continue
        hit = bool(row["correct"])
        if hit and not control[row["id"]]:
            wins += 1
        elif not hit and control[row["id"]]:
            losses += 1
    return {"wins": wins, "losses": losses, "p": exact_mcnemar(wins, losses)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--heldout-items", required=True)
    parser.add_argument("--base-probs", required=True)
    parser.add_argument("--heldout-ids", required=True)
    parser.add_argument("--control-preds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    heldout_payload = json.load(open(args.heldout_ids))
    heldout_ids = set(heldout_payload["ids"])

    jev_arm = [
        json.loads(line)
        for line in open(f"{args.seed_dir}/jevbench.pred.jsonl")
        if line.strip()
    ]
    jev_control = [
        json.loads(line) for line in open(args.control_preds) if line.strip()
    ]
    held_arm = [
        json.loads(line)
        for line in open(f"{args.seed_dir}/heldout.pred.jsonl")
        if line.strip()
    ]
    held_base_rows = list(
        load_heldout_base(args.heldout_items, args.base_probs, heldout_ids).values()
    )
    # The arm dump includes the 170 abstention rows with no label index and a
    # forced-wrong ``correct``.  Both columns must be the same 6,254 items, or
    # the delta compares an arm that counts them against a base that does not.
    base_ids = {row["id"] for row in held_base_rows}
    held_arm_scored = [row for row in held_arm if row["id"] in base_ids]
    heldout_items = [
        json.loads(line) for line in open(args.heldout_items) if line.strip()
    ]
    held_frac, held_score = fractional_score_rows(heldout_items)
    jev_frac, jev_score = fractional_score_rows(
        [
            {"question": item.question, "label": item.label}
            for item in load_jevbench()
        ]
    )

    jev_metrics = summarize(jev_arm)
    jev_pair = paired(jev_arm, jev_control)
    jev_admissible = jev_metrics["cc"] >= 0.5187 and not (
        jev_pair["losses"] > jev_pair["wins"] and jev_pair["p"] < 0.05
    )

    held_metrics = summarize(held_arm_scored)
    base_metrics = summarize(held_base_rows)
    held_pair = paired(held_arm, held_base_rows)
    held_delta = held_metrics["cc"] - base_metrics["cc"]
    promoted = jev_admissible and held_pair["wins"] > held_pair["losses"] and held_pair["p"] < 0.05

    result = {
        "seed_dir": args.seed_dir,
        "jevbench": {
            "metrics": jev_metrics,
            "paired_vs_control": jev_pair,
            "control_cc": 0.5187,
            "admissible": jev_admissible,
        },
        "heldout": {
            "metrics": held_metrics,
            "base_metrics": base_metrics,
            "delta_cc": held_delta,
            "paired_vs_base": held_pair,
            "significant": held_pair["wins"] > held_pair["losses"] and held_pair["p"] < 0.05,
        },
        "promoted": promoted,
        "scope": {
            "heldout_frozen_rows": len(heldout_ids),
            "heldout_evaluated_rows": held_metrics["n"],
            "heldout_unscorable_abstention_rows": len(heldout_ids) - held_metrics["n"],
            "heldout_arm_rows_dropped_unscorable": len(held_arm) - len(held_arm_scored),
            "internal_validation_note": (
                "Internal validation numbers are not the gate. That split is a "
                "different and harder population than JevBench (choice about "
                "0.52 against about 0.60 on the 20K proxy), and its n is 47,038 "
                "of 47,389 because 351 abstention rows have no label index. Do "
                "not compare the two in one table without this warning."
            ),
            "abstention_not_gated": (
                "Standard C measures task accuracy only and no instrument gates "
                "abstention: JevBench has no abstention rows, and the frozen "
                "held-out abstention rows are unscorable for lack of a label "
                "index. A promoted arm is not evidence that abstention works; "
                "#232/#254 (answerable head) must add an abstention metric to "
                "the acceptance path."
            ),
        },
        "score_label_rounding": {
            "method": "score gold is int(round(float(label)))",
            "rps_definition_note": (
                "RPS is not computed by this gate and its count is zero in both "
                "instruments. If it were computed today, main's RPS uses the "
                "rounded label index while score_mae in the same loop uses the "
                "true fractional label; krino-ab's #253 follow-up fixes the RPS "
                "target distribution. A later RPS is therefore not comparable "
                "to one this gate would have produced."
            ),
            "heldout_score_rows": held_score,
            "heldout_fractional_score_rows": held_frac,
            "jevbench_score_rows": jev_score,
            "jevbench_fractional_score_rows": jev_frac,
        },
    }
    atomic_dump(result, args.output)
    print(json.dumps(result, indent=1))
    print(
        f"JevBench cc {jev_metrics['cc']:.4f} (control 0.5187, delta "
        f"{jev_metrics['cc'] - 0.5187:+.4f}); held-out cc {held_metrics['cc']:.4f} "
        f"vs base {base_metrics['cc']:.4f} (delta {held_delta:+.4f}); "
        f"held-out n {held_metrics['n']} of {len(heldout_ids)} "
        f"({len(heldout_ids) - held_metrics['n']} abstention rows unscorable); "
        f"admissible {jev_admissible}; promoted {promoted}"
    )
    print(
        "abstention is not gated by standard C; see scope.abstention_not_gated "
        "and #232/#254"
    )


if __name__ == "__main__":
    main()
