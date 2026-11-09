"""Sweep the cascade threshold on dev, then evaluate JevBench once (#342).

The threshold is selected on the development set by maximising
``accuracy / (1 + 2 * f)``, where ``f`` is the fraction of items that trigger the
second pass; ties go to the largest threshold.  JevBench is evaluated once, at
the selected threshold, and never used to choose or adjust it.

Stability is checked on dev with two declared split halves: each half selects a
threshold on its own and is then scored on the other half.  The issue's third
reading applies if that disagrees.

Usage::

    python model/scripts/sweep_cascade.py --dev pairs_dev.json --jevbench pairs_jev.json \\
        --dev-panel2 panel2_dev.json --jevbench-panel2 panel2_jev.json \\
        --output results/jevbench/cascade_F_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.jevbench import tier_from_id  # noqa: E402
from model.evaluation.cascade import (  # noqa: E402
    best,
    evaluate,
    prediction,
    split_halves,
    sweep,
    threshold_grid,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quality_quantiles(rows: list[dict], quantiles=(0.1, 0.5, 0.9)) -> dict:
    values = sorted(row["margin"] for row in rows)
    if not values:
        return {}
    out = {}
    for q in quantiles:
        out[str(q)] = values[min(len(values) - 1, int(q * len(values)))]
    return out


def mcnemar(first: dict[str, bool], second: dict[str, bool], ids: list[str]) -> dict:
    first_only = sum(1 for i in ids if first[i] and not second[i])
    second_only = sum(1 for i in ids if second[i] and not first[i])
    n = first_only + second_only
    if n == 0:
        p = 1.0
    else:
        k = min(first_only, second_only)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * (0.5**n))
    return {
        "cascade_only_correct": first_only,
        "base_only_correct": second_only,
        "discordant": n,
        "exact_p": p,
    }


def answers(rows: list[dict], threshold: float) -> dict[str, int]:
    out = {}
    for row in rows:
        answer, _ = prediction(
            row["base"],
            row["candidates"],
            row["scores"],
            row["margin"],
            threshold if row["eligible"] else 0.0,
        )
        out[row["id"]] = answer
    return out


def mcnemar_block(rows: list[dict], threshold: float) -> dict:
    chosen = answers(rows, threshold)
    first = {row["id"]: chosen[row["id"]] == row["gold"] for row in rows}
    second = {row["id"]: row["base"] == row["gold"] for row in rows}
    ids = sorted(first)
    return mcnemar(first, second, ids)


def by_tier(rows: list[dict], threshold: float, tiers: dict[str, str]) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        tier = tiers.get(row["id"], "unknown")
        cell = out.setdefault(
            tier, {"items": 0, "correct": 0, "base_correct": 0, "triggered": 0}
        )
        answer, spent = prediction(
            row["base"],
            row["candidates"],
            row["scores"],
            row["margin"],
            threshold if row["eligible"] else 0.0,
        )
        cell["items"] += 1
        cell["correct"] += int(answer == row["gold"])
        cell["base_correct"] += int(row["base"] == row["gold"])
        cell["triggered"] += int(spent)
    for cell in out.values():
        cell["accuracy"] = cell["correct"] / cell["items"]
        cell["base_accuracy"] = cell["base_correct"] / cell["items"]
        cell["trigger_fraction"] = cell["triggered"] / cell["items"]
    return out


def latency_block(
    panel2_rows: dict[str, float], pairs: list[dict], fraction: float
) -> dict:
    base = sum(panel2_rows.values()) / len(panel2_rows) if panel2_rows else 0.0
    eligible = [row["latency_s"] for row in pairs if row["eligible"]]
    second = sum(eligible) / len(eligible) if eligible else 0.0
    return {
        "first_pass_mean_s": base,
        "second_pass_mean_s": second,
        "trigger_fraction": fraction,
        "implied_mean_s": base + fraction * second,
        "mean_forwards": 1.0 + 2.0 * fraction,
    }


def recorded_path(path: Path) -> str:
    """A path safe to commit: repo-relative inside the repo, else a placeholder."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{Path(path).name}"


def normalize(rows: list[dict]) -> list[dict]:
    """JSON stores dict keys as strings; the rule looks options up by index."""
    for row in rows:
        row["scores"] = {int(key): value for key, value in row["scores"].items()}
        row["candidates"] = [int(index) for index in row["candidates"]]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--jevbench", type=Path, required=True)
    parser.add_argument("--dev-panel2", type=Path, required=True)
    parser.add_argument("--jevbench-panel2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    args = parser.parse_args()

    dev = load(args.dev)
    jev = load(args.jevbench)
    dev_rows = normalize(dev["rows"])
    jev_rows = normalize(jev["rows"])
    dev_panel2 = {
        r["id"]: r["latency_s"] for r in load(args.dev_panel2)["predictions_natural"]
    }
    jev_panel2 = {
        r["id"]: r["latency_s"]
        for r in load(args.jevbench_panel2)["predictions_natural"]
    }

    config = {
        "arm": "F",
        "base_readout": "panel 2, natural order, shipped option order, default attention, "
        "tie via choose_option (first maximum by natural index)",
        "second_readout": {
            "template": "the judgement template of model/evaluation/judgement_prompt.py",
            "norm": "mean",
            "strategy": "description",
            "scores_compared": "log P(' yes') per candidate",
            "tie_break": "smaller natural option index",
            "noul_index_map": "inverted (panel 2 no/yes; judgement true/false first)",
            "unavailable_items": "a noul item without criteria keeps the first-pass answer",
        },
        "margin": "softmax over the K code logits, p1 - p2",
        "trigger": "margin < threshold",
        "objective": "accuracy / (1 + 2 * trigger_fraction)",
        "objective_tie_break": "largest threshold",
        "threshold_grid": "0, every distinct dev margin, 1",
        "split_modes": ["even_odd", "first_second"],
        "dev_items": len(dev_rows),
        "jevbench_items": len(jev_rows),
    }
    # The hash covers the recipe, so it can be fixed before any model run; the
    # input digests are recorded beside it because they only exist afterwards.
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    input_digests = {
        "dev_pairs_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
        "jevbench_pairs_sha256": hashlib.sha256(args.jevbench.read_bytes()).hexdigest(),
        "dev_panel2_sha256": hashlib.sha256(args.dev_panel2.read_bytes()).hexdigest(),
        "jevbench_panel2_sha256": hashlib.sha256(
            args.jevbench_panel2.read_bytes()
        ).hexdigest(),
    }
    config["config_hash"] = config_hash
    config["input_digests"] = input_digests
    print(f"config_hash {config_hash}", flush=True)
    print(json.dumps(config, sort_keys=True), flush=True)
    if args.print_config_hash:
        return 0

    grid = threshold_grid(dev_rows)
    curve = sweep(dev_rows, grid)
    selected = best(curve)

    halves = {}
    for mode in ("even_odd", "first_second"):
        left, right = split_halves([row["id"] for row in dev_rows], mode)
        by_id = {row["id"]: row for row in dev_rows}
        left_rows = [by_id[i] for i in left]
        right_rows = [by_id[i] for i in right]
        chosen_left = best(sweep(left_rows, threshold_grid(left_rows)))
        chosen_right = best(sweep(right_rows, threshold_grid(right_rows)))
        halves[mode] = {
            "left_threshold": chosen_left["threshold"],
            "right_threshold": chosen_right["threshold"],
            "agree": chosen_left["threshold"] == chosen_right["threshold"],
            "left_on_right": evaluate(right_rows, chosen_left["threshold"]),
            "right_on_left": evaluate(left_rows, chosen_right["threshold"]),
        }

    jev_result = evaluate(jev_rows, selected["threshold"])
    result = {
        "_provenance": {
            "arm": "F",
            "config_hash": config_hash,
            "dev": recorded_path(args.dev),
            "jevbench": recorded_path(args.jevbench),
            "dev_panel2": recorded_path(args.dev_panel2),
            "jevbench_panel2": recorded_path(args.jevbench_panel2),
        },
        "config": config,
        "dev": {
            "curve": curve,
            "selected": selected,
            "base": evaluate(dev_rows, 0.0),
            "mcnemar_cascade_vs_base": mcnemar_block(dev_rows, selected["threshold"]),
            "margin_quantiles": quality_quantiles(dev_rows),
            "latency": latency_block(
                dev_panel2, dev_rows, selected["trigger_fraction"]
            ),
        },
        "split_half": halves,
        "jevbench": {
            "selected_threshold": selected["threshold"],
            "result": jev_result,
            "base": evaluate(jev_rows, 0.0),
            "mcnemar_cascade_vs_base": mcnemar_block(jev_rows, selected["threshold"]),
            "margin_quantiles": quality_quantiles(jev_rows),
            "by_tier": {},
            "latency": latency_block(
                jev_panel2, jev_rows, selected["trigger_fraction"]
            ),
        },
    }
    # Tier labels are not in the pairs file; the JevBench id carries the tier.
    tiers = {row["id"]: tier_from_id(row["id"].split(":", 1)[-1]) for row in jev_rows}
    result["jevbench"]["by_tier"] = by_tier(jev_rows, selected["threshold"], tiers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"dev selected theta {selected['threshold']:.6f} "
        f"acc {selected['accuracy']:.4f} f {selected['trigger_fraction']:.4f} "
        f"objective {selected['objective']:.4f}",
        flush=True,
    )
    print(
        f"jevbench acc {jev_result['accuracy']:.4f} ({jev_result['correct']}/{jev_result['items']}) "
        f"f {jev_result['trigger_fraction']:.4f} forwards {jev_result['mean_forwards']:.3f}",
        flush=True,
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
