#!/usr/bin/env python3
"""Paired McNemar tests and the gate screen between JevBench prediction dumps.

The hard tier is 111 items. An unpaired standard error is about 4.7 points, which
is large enough that a 4-point difference between two models can be one model
bettering the other on a handful of the same items -- or on different items, which
is a weaker claim. This compares the dumps item by item.

Each pair prints the three-state interval screen from ``research/37-gate-policy.md``
beside the exact test: the first dump of a pair is read as the arm and the second
as its control. The chance-corrected interval, the detectable difference and the
states are printed together, so a null is never read on the p-value alone.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.gate import PairedOutcome, paired_screen


def _records(path: Path) -> list[dict]:
    """Prediction records from a dump, whichever of the two shapes it uses.

    Accepts a JSON object with a ``predictions`` array, or JSONL with one
    prediction per line (what ``eval_gate.py --predictions-out`` writes).
    """
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)["predictions"]
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    return payload


def load(path: Path) -> dict[str, bool]:
    """Per-item correctness from a prediction dump."""
    return {row["id"]: bool(row["correct"]) for row in _records(path)}


def load_outcomes(path: Path) -> dict[str, PairedOutcome]:
    """Per-item correctness and option count, for the gate screen.

    The chance level belongs to the item, so the screen needs the option count
    as well as the outcome. ``eval_gate.py`` writes ``n_options``; older dumps
    that carry only a probability or key list fall back to its length.
    """
    outcomes: dict[str, PairedOutcome] = {}
    for row in _records(path):
        n_options = row.get("n_options")
        if n_options is None:
            for key in ("probs", "keys", "logits", "logp_yes"):
                value = row.get(key)
                if not isinstance(value, list) or not value:
                    continue
                # A criteria-less noul item records one signed logit but still
                # has two options ("no", "yes").
                if len(value) == 1 and row.get("q_type") == "noul":
                    n_options = 2
                else:
                    n_options = len(value)
                break
        if n_options is None or int(n_options) < 2:
            raise ValueError(f"{path}: no usable option count for id {row['id']!r}")
        outcomes[row["id"]] = PairedOutcome(
            correct=bool(row["correct"]), n_options=int(n_options)
        )
    return outcomes


def mcnemar(first: dict[str, bool], second: dict[str, bool], ids: list[str]):
    first_only = sum(1 for i in ids if first[i] and not second[i])
    second_only = sum(1 for i in ids if second[i] and not first[i])
    n = first_only + second_only
    if n == 0:
        return first_only, second_only, 1.0
    k = min(first_only, second_only)
    p = sum(comb(n, i) for i in range(k + 1)) * (0.5**n)
    return first_only, second_only, min(1.0, 2 * p)


def main() -> int:
    tier = sys.argv[1] if len(sys.argv) > 1 else "hard"
    runs = [arg.split("=", 1) for arg in sys.argv[2:]]
    outcomes = {name: load_outcomes(Path(path)) for name, path in runs}
    data = {
        name: {item_id: row.correct for item_id, row in rows.items()}
        for name, rows in outcomes.items()
    }
    names = list(data)
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    ids = [i for i in ids if str(i).startswith(tier)]
    print(f"{tier} items shared by every dump: {len(ids)}")
    for name in names:
        acc = sum(data[name][i] for i in ids) / len(ids)
        print(f"  {name:<16} {acc:.4f}  ({sum(data[name][i] for i in ids)}/{len(ids)})")
    print()
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            screen = paired_screen(outcomes[first], outcomes[second], ids)
            print(
                f"{first} vs {second}: +{screen.wins} / -{screen.losses} discordant, "
                f"McNemar p={screen.p_value:.3f}"
            )
            detectable = (
                "not estimable (no discordant pairs)"
                if screen.detectable_pp is None
                else f"{screen.detectable_pp:.1f} pp"
            )
            print(
                f"  chance-corrected delta {screen.delta_pp:+.2f} pp, "
                f"95% CI [{screen.ci_low_pp:+.2f}, {screen.ci_high_pp:+.2f}] pp"
            )
            print(
                f"  detectable difference {detectable} at n={screen.n}, "
                f"discordance {screen.discordance:.3f}"
            )
            print(f"  states: {', '.join(sorted(screen.states))}")
            print(f"  action: {screen.action} - {screen.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
