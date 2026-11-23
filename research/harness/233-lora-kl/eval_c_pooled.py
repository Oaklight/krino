"""Pooled standard C, dual gate over three seeds.

Per held-out item, average the per-item chance-corrected value over seeds; the
paired McNemar uses the majority outcome (mean hit > 0.5) against the frozen
base.  JevBench is pooled the same way against the control.  Both numbers are
reported side by side; promotion requires JevBench admissible and a significant
held-out improvement.
"""

import argparse
import json
import math


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def corrected(hit, n_options):
    return (float(bool(hit)) - 1.0 / n_options) / (1.0 - 1.0 / n_options)


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


def label_index(question, label):
    q_type = question["type"]
    if q_type == "noul":
        return 1 if bool(label) else 0
    if q_type == "choice":
        keys = list(question.get("criteria", {}))
        return keys.index(label) if label in keys else -1
    return int(round(float(label)))


def load_base(items_path, base_probs_path, heldout_ids):
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


def pool(seed_dirs, filename):
    per_id = {}
    for directory in seed_dirs:
        rows = [
            json.loads(line)
            for line in open(f"{directory}/{filename}")
            if line.strip()
        ]
        for row in rows:
            entry = per_id.setdefault(
                row["id"],
                {"q_type": row.get("q_type"), "n_options": row["n_options"], "hits": [], "probs": []},
            )
            entry["hits"].append(bool(row["correct"]))
            if row.get("probs"):
                entry["probs"].append(row["probs"])
    pooled = []
    for item_id, entry in per_id.items():
        mean_hit = sum(entry["hits"]) / len(entry["hits"])
        probs = (
            [sum(values) / len(values) for values in zip(*entry["probs"])]
            if entry["probs"]
            else [0.0] * entry["n_options"]
        )
        pooled.append(
            {
                "id": item_id,
                "q_type": entry["q_type"],
                "n_options": entry["n_options"],
                "mean_hit": mean_hit,
                "correct": mean_hit > 0.5,
                "probs": probs,
            }
        )
    return pooled


def metrics(rows):
    usable = [row for row in rows if row.get("n_options", 0) > 1]
    if not usable:
        return {"n": 0}
    values = [corrected(row["mean_hit"], row["n_options"]) for row in usable]
    per_type = {}
    for q_type in ("noul", "choice", "score"):
        subset = [row for row in usable if row.get("q_type") == q_type]
        if subset:
            per_type[q_type] = {
                "n": len(subset),
                "cc": sum(corrected(row["mean_hit"], row["n_options"]) for row in subset)
                / len(subset),
            }
    return {
        "n": len(usable),
        "cc": sum(values) / len(values),
        "ece": ece(usable),
        "by_type": per_type,
    }


def paired(rows, control):
    wins = losses = 0
    for row in rows:
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
    parser.add_argument("--seed-dirs", nargs="+", required=True)
    parser.add_argument("--heldout-items", required=True)
    parser.add_argument("--base-probs", required=True)
    parser.add_argument("--heldout-ids", required=True)
    parser.add_argument("--control-preds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    heldout_ids = set(json.load(open(args.heldout_ids))["ids"])
    jev_pooled = pool(args.seed_dirs, "jevbench.pred.jsonl")
    held_pooled = pool(args.seed_dirs, "heldout.pred.jsonl")
    control_rows = [
        json.loads(line) for line in open(args.control_preds) if line.strip()
    ]
    control = {row["id"]: bool(row["correct"]) for row in control_rows}
    base_rows = list(load_base(args.heldout_items, args.base_probs, heldout_ids).values())
    base = {row["id"]: bool(row["correct"]) for row in base_rows}

    jev_metrics = metrics(jev_pooled)
    jev_pair = paired(jev_pooled, control)
    jev_admissible = jev_metrics["cc"] >= 0.5187 and not (
        jev_pair["losses"] > jev_pair["wins"] and jev_pair["p"] < 0.05
    )
    held_metrics = metrics(held_pooled)
    base_metrics = {
        "n": len(base_rows),
        "cc": sum(corrected(row["correct"], row["n_options"]) for row in base_rows)
        / len(base_rows),
    }
    held_pair = paired(held_pooled, base)
    held_delta = held_metrics["cc"] - base_metrics["cc"]
    promoted = jev_admissible and held_pair["wins"] > held_pair["losses"] and held_pair["p"] < 0.05

    result = {
        "seeds": args.seed_dirs,
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
    }
    json.dump(result, open(args.output, "w"), indent=1)
    print(json.dumps(result, indent=1))
    print(
        f"pooled JevBench cc {jev_metrics['cc']:.4f} (control 0.5187); "
        f"pooled held-out cc {held_metrics['cc']:.4f} vs base {base_metrics['cc']:.4f} "
        f"(delta {held_delta:+.4f}); admissible {jev_admissible}; promoted {promoted}"
    )


if __name__ == "__main__":
    main()
