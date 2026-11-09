"""Evaluate the set head on JevBench and the development set (#340, arm D).

Reads the cached option states for one split in both orders, runs the trained
head, and reports the four numbers the arm is judged on -- natural accuracy,
reversed accuracy, their difference, and the fraction of items whose choice
changes -- with an exact paired McNemar on the natural/reversed pair.

When ``--panel2`` is given, the same items are compared against the shipped
panel-2 file: accuracy in both orders, the discordant counts and the exact
p-value, and panel 2's own order sensitivity beside the head's.

Usage::

    python model/scripts/eval_set_head.py --cache <cache> --head runs/set_head/head.pt \\
        --split jevbench --rows rows_jevbench.jsonl \\
        --panel2 results/jevbench/panel2_Qwen_Qwen3-Reranker-4B.json \\
        --output results/jevbench/set_head_jevbench_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.jevbench import tier_from_id  # noqa: E402
from model.evaluation.panel2 import has_exact_tie  # noqa: E402
from model.src.set_head import SetHead  # noqa: E402
from train_set_head import EVAL_BATCH, load_cache  # noqa: E402


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


@torch.inference_mode()
def score_order(
    cache: Path, split: str, order: str, head: SetHead, device: str
) -> dict:
    """Per-item prediction, gold probability and tie flag for one order."""
    head.eval()
    groups = load_cache(cache, split, order)
    scored: dict[str, dict] = {}
    for k in sorted(groups):
        entry = groups[k]
        states, gold, ids = entry["states"], entry["gold"], entry["ids"]
        for start in range(0, states.shape[0], EVAL_BATCH):
            stop = min(start + EVAL_BATCH, states.shape[0])
            indices = torch.arange(start, stop)
            selected = states[indices].to(device=device, dtype=torch.float32)
            valid = torch.ones(selected.shape[:2], dtype=torch.bool, device=device)
            probabilities = head.probabilities(selected, valid).cpu().numpy()
            targets = gold[indices].numpy()
            for row, item_id in enumerate(ids[start:stop]):
                vector = probabilities[row].astype(np.float64)
                prediction = int(np.argmax(vector))  # first maximum by natural index
                scored[item_id] = {
                    "pred": prediction,
                    "gold": int(targets[row]),
                    "correct": prediction == int(targets[row]),
                    "p_gold": float(vector[int(targets[row])]),
                    "tie": bool(has_exact_tie(vector.tolist())),
                }
    return scored


def accuracy_by(scored: dict, key, rows: dict) -> dict:
    """Accuracy grouped by a label taken from the row or computed from the id."""
    groups: dict[str, list[bool]] = {}
    for item_id, item in scored.items():
        label = key(item_id, rows.get(item_id, {}))
        groups.setdefault(label, []).append(item["correct"])
    return {
        label: {
            "correct": int(sum(values)),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for label, values in sorted(groups.items())
    }


def mcnemar(first: dict[str, bool], second: dict[str, bool], ids: list[str]) -> dict:
    """Exact paired McNemar, as in ``paired_mcnemar.py``."""
    from math import comb

    first_only = sum(1 for i in ids if first[i] and not second[i])
    second_only = sum(1 for i in ids if second[i] and not first[i])
    n = first_only + second_only
    if n == 0:
        p = 1.0
    else:
        k = min(first_only, second_only)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * (0.5**n))
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": n,
        "exact_p": p,
    }


def _screen(
    ours: dict[str, bool],
    theirs: dict[str, bool],
    options: dict[str, int],
    ids: list[str],
) -> dict:
    """The project's three-state interval screen, arm = the set head."""
    from model.evaluation.gate import PairedOutcome, paired_screen

    screen = paired_screen(
        {item: PairedOutcome(ours[item], options[item]) for item in ids},
        {item: PairedOutcome(theirs[item], options[item]) for item in ids},
        ids,
    )
    return {
        "n": screen.n,
        "delta_pp": screen.delta_pp,
        "ci_low_pp": screen.ci_low_pp,
        "ci_high_pp": screen.ci_high_pp,
        "wins": screen.wins,
        "losses": screen.losses,
        "p_value": screen.p_value,
        "discordance": screen.discordance,
        "detectable_pp": screen.detectable_pp,
        "detectable_cc_pp": screen.detectable_cc_pp,
        "states": sorted(screen.states),
        "action": screen.action,
        "reason": screen.reason,
    }


def order_summary(natural: dict, reversed_order: dict) -> dict:
    ids = sorted(set(natural) & set(reversed_order))
    if not ids:
        raise SystemExit("no shared ids between the two orders")
    changed = sum(1 for i in ids if natural[i]["pred"] != reversed_order[i]["pred"])
    first = {i: natural[i]["correct"] for i in ids}
    second = {i: reversed_order[i]["correct"] for i in ids}
    acc_natural = sum(first.values()) / len(ids)
    acc_reversed = sum(second.values()) / len(ids)
    return {
        "items": len(ids),
        "accuracy_natural": acc_natural,
        "accuracy_reversed": acc_reversed,
        "difference_pp": (acc_natural - acc_reversed) * 100.0,
        "changed_fraction": changed / len(ids),
        "changed_items": changed,
        "mcnemar_natural_vs_reversed": mcnemar(first, second, ids),
        "tie_items_natural": int(sum(natural[i]["tie"] for i in ids)),
        "tie_items_reversed": int(sum(reversed_order[i]["tie"] for i in ids)),
    }


def against_panel2(
    natural: dict, reversed_order: dict, panel2: dict, max_options: int | None = None
) -> dict:
    """D against the shipped panel-2 file, keyed on the JevBench item id.

    ``max_options`` restricts the comparison to items the shipped 26-letter code
    alphabet can read; the development cell also holds 80-option banking77
    items, whose panel-2 reference was read with the extended alphabet.
    """

    def raw(item_id: str) -> str:
        return item_id.split(":", 1)[1] if ":" in item_id else item_id

    natural = {raw(item_id): item for item_id, item in natural.items()}
    reversed_order = {raw(item_id): item for item_id, item in reversed_order.items()}
    reference = {
        "natural": {raw(row["id"]): row for row in panel2["predictions_natural"]},
        "reversed": {raw(row["id"]): row for row in panel2["predictions_reversed"]},
    }
    if max_options is not None:
        keep = {
            item_id
            for item_id, row in reference["natural"].items()
            if int(row["n_options"]) <= max_options
        }
        natural = {k: v for k, v in natural.items() if k in keep}
        reversed_order = {k: v for k, v in reversed_order.items() if k in keep}
        reference = {
            label: {k: v for k, v in rows.items() if k in keep}
            for label, rows in reference.items()
        }
    result = {}
    for label, scored in (("natural", natural), ("reversed", reversed_order)):
        shared = sorted(set(scored) & set(reference[label]))
        ours = {i: scored[i]["correct"] for i in shared}
        theirs = {i: bool(reference[label][i]["correct"]) for i in shared}
        result[label] = {
            "items": len(shared),
            "set_head_correct": sum(ours.values()),
            "panel2_correct": sum(theirs.values()),
            "set_head_accuracy": sum(ours.values()) / len(shared),
            "panel2_accuracy": sum(theirs.values()) / len(shared),
            "difference_pp": (sum(ours.values()) - sum(theirs.values()))
            / len(shared)
            * 100.0,
            "mcnemar_set_head_vs_panel2": mcnemar(ours, theirs, shared),
            "screen": _screen(
                ours,
                theirs,
                {i: int(reference[label][i]["n_options"]) for i in shared},
                shared,
            ),
        }
    shared = sorted(set(natural) & set(reversed_order) & set(reference["natural"]))

    def natural_position(row: dict, order: str) -> int:
        """Panel 2 records a reversed row in display coordinates."""
        if order == "natural":
            return int(row["pred_pos"])
        return int(row["n_options"]) - 1 - int(row["pred_pos"])

    p2_changed = sum(
        1
        for i in shared
        if natural_position(reference["natural"][i], "natural")
        != natural_position(reference["reversed"][i], "reversed")
    )
    result["order_sensitivity"] = {
        "items": len(shared),
        "set_head_changed_fraction": sum(
            1 for i in shared if natural[i]["pred"] != reversed_order[i]["pred"]
        )
        / len(shared),
        "panel2_changed_fraction": p2_changed / len(shared),
    }
    return result


def summarize(scored: dict, rows: dict) -> dict:
    ids = sorted(scored)
    correct = sum(scored[i]["correct"] for i in ids)

    def kind(item_id: str, row: dict) -> str:
        return str(row.get("type", "unknown"))

    def tier(item_id: str, row: dict) -> str:
        raw = item_id.split(":", 1)[1] if ":" in item_id else item_id
        return tier_from_id(raw)

    return {
        "items": len(ids),
        "correct": int(correct),
        "accuracy": correct / len(ids),
        "by_type": accuracy_by(scored, kind, rows),
        "by_tier": accuracy_by(scored, tier, rows),
        "mean_gold_probability": float(np.mean([scored[i]["p_gold"] for i in ids])),
        "tie_items": int(sum(scored[i]["tie"] for i in ids)),
    }


def recorded_path(path: Path) -> str:
    """A path safe to commit: repo-relative inside the repo, else a placeholder."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{Path(path).name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("jevbench", "dev", "test"))
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--panel2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.head, map_location="cpu", weights_only=True)
    config = payload["config"]
    head = SetHead(
        config["hidden_size"],
        width=config["width"],
        layers=config["layers"],
        heads=config["heads"],
        ffn=config["ffn"],
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device)

    rows = load_rows(args.rows)
    natural = score_order(args.cache, args.split, "natural", head, device)
    reversed_order = score_order(args.cache, args.split, "reversed", head, device)

    result = {
        "_provenance": {
            "head": recorded_path(args.head),
            "head_sha256": hashlib.sha256(args.head.read_bytes()).hexdigest(),
            "cache": recorded_path(args.cache),
            "split": args.split,
            "rows": recorded_path(args.rows),
            "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
            "torch_version": torch.__version__,
        },
        "config": config,
        "natural": summarize(natural, rows),
        "reversed": summarize(reversed_order, rows),
        "order": order_summary(natural, reversed_order),
        "predictions_natural": [
            {"id": item_id, **natural[item_id]} for item_id in sorted(natural)
        ],
        "predictions_reversed": [
            {"id": item_id, **reversed_order[item_id]}
            for item_id in sorted(reversed_order)
        ],
    }
    if args.panel2:
        panel2 = json.loads(args.panel2.read_text(encoding="utf-8"))
        result["panel2"] = {
            "path": recorded_path(args.panel2),
            "sha256": hashlib.sha256(args.panel2.read_bytes()).hexdigest(),
        }
        result["against_panel2"] = against_panel2(natural, reversed_order, panel2)
        if any(int(row["n_options"]) > 26 for row in panel2["predictions_natural"]):
            result["against_panel2_code_range"] = against_panel2(
                natural, reversed_order, panel2, max_options=26
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    order = result["order"]
    print(
        f"{args.split}: natural {order['accuracy_natural']:.4f} "
        f"reversed {order['accuracy_reversed']:.4f} "
        f"delta {order['difference_pp']:+.2f} pp "
        f"changed {order['changed_fraction']:.4f} "
        f"McNemar +{order['mcnemar_natural_vs_reversed']['first_only_correct']} "
        f"-{order['mcnemar_natural_vs_reversed']['second_only_correct']} "
        f"p={order['mcnemar_natural_vs_reversed']['exact_p']:.4g}",
        flush=True,
    )
    if "against_panel2" in result:
        for label in ("natural", "reversed"):
            cell = result["against_panel2"][label]
            print(
                f"  vs panel2 {label}: {cell['set_head_correct']} vs {cell['panel2_correct']} "
                f"of {cell['items']} ({cell['difference_pp']:+.2f} pp) "
                f"+{cell['mcnemar_set_head_vs_panel2']['first_only_correct']} "
                f"-{cell['mcnemar_set_head_vs_panel2']['second_only_correct']} "
                f"p={cell['mcnemar_set_head_vs_panel2']['exact_p']:.4g}",
                flush=True,
            )
        sensitivity = result["against_panel2"]["order_sensitivity"]
        print(
            f"  changed fraction: set head {sensitivity['set_head_changed_fraction']:.4f} "
            f"vs panel2 {sensitivity['panel2_changed_fraction']:.4f}",
            flush=True,
        )
        if "against_panel2_code_range" in result:
            subset = result["against_panel2_code_range"]["natural"]
            print(
                f"  within the 26-code range: {subset['set_head_correct']} vs "
                f"{subset['panel2_correct']} of {subset['items']} "
                f"({subset['difference_pp']:+.2f} pp)",
                flush=True,
            )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
