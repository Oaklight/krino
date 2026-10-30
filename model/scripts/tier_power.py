#!/usr/bin/env python3
"""How large an effect a paired item set can detect, before any model runs.

`research/21` and `research/13` both end on the same sentence: the hard tier
cannot resolve the difference being asked about. This script turns that into
arithmetic. Given an item count and the rate at which two comparable systems
disagree on it, `model.evaluation.paired_power` gives the exact power of a
two-sided McNemar test at any accuracy difference, the smallest difference the
set detects, and the item count a larger set would need.

The discordance rate is the only input that has to be measured, and it comes
from the same prediction dumps `model/scripts/paired_mcnemar.py` reads:

    python model/scripts/tier_power.py \\
        --preds tower=model/experiments/jevbench_tower_ep10_preds.json \\
                xenc=model/experiments/jevbench_xenc_ep3_preds.json \\
        --tier hard --target-power 0.8

or, when the rate is already known:

    python model/scripts/tier_power.py --items 111 --discordance 0.324
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.paired_power import (  # noqa: E402
    detectable_delta,
    items_for_power,
    power,
)

DEFAULT_DELTAS = "0.02,0.05,0.10,0.15,0.20"


def _load_dump(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    return {row["id"]: bool(row["correct"]) for row in payload}


def _discordance(runs: dict[str, dict[str, bool]], tier: str) -> tuple[float, int, int]:
    """Discordance over the items every dump shares, restricted to ``tier``.

    Returns the rate, the number of items and the number of discordant pairs.
    """
    ids = set.intersection(*(set(dump) for dump in runs.values()))
    if tier:
        ids = {item_id for item_id in ids if str(item_id).startswith(tier)}
    names = list(runs)
    discordant = 0
    for item_id in ids:
        outcomes = {runs[name][item_id] for name in names}
        if len(outcomes) > 1:
            discordant += 1
    if not ids:
        raise ValueError(f"no shared items{f' in tier {tier!r}' if tier else ''}")
    return discordant / len(ids), len(ids), discordant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", nargs="*", default=[], metavar="NAME=PATH")
    parser.add_argument("--tier", default="", help="id prefix to restrict to, e.g. hard")
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--discordant", type=int, default=None)
    parser.add_argument("--discordance", type=float, default=None)
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    source = None
    if args.preds:
        runs = {}
        for entry in args.preds:
            name, _, path = entry.partition("=")
            if not path:
                parser.error(f"--preds entry must be NAME=PATH, got {entry!r}")
            runs[name] = _load_dump(Path(path))
        if len(runs) == 1:
            parser.error("--preds needs at least two dumps to measure a discordance")
        rate, n_items, n_discordant = _discordance(runs, args.tier)
        source = "prediction dumps"
    else:
        if args.items is None:
            parser.error("pass --items (with --discordant or --discordance) or --preds")
        n_items = args.items
        if args.discordant is not None:
            rate = args.discordant / n_items
        elif args.discordance is not None:
            rate = args.discordance
        else:
            parser.error("pass --discordant or --discordance")
        n_discordant = round(rate * n_items)
        source = "supplied"

    print("=== Paired McNemar power ===", flush=True)
    print(f"source: {source}", flush=True)
    if args.preds:
        print(f"dumps:  {', '.join(name for name, _ in (e.split('=', 1) for e in args.preds))}",
              flush=True)
        print(f"tier:   {args.tier or '(all shared items)'}", flush=True)
    print(f"items: {n_items}, discordant: {n_discordant}, "
          f"rate: {rate:.4f}, alpha: {args.alpha}", flush=True)

    deltas = [float(part) for part in args.deltas.split(",") if part.strip()]
    by_delta: dict[str, float] = {}
    print(f"\npower at this size (target {args.target_power:.0%}):", flush=True)
    for delta in deltas:
        value = power(n_items, rate, delta, args.alpha)
        by_delta[f"{delta:g}"] = value
        flag = "  <- detected" if value >= args.target_power else ""
        print(f"  delta {delta * 100:5.1f} pp -> power {value:.3f}{flag}", flush=True)

    detectable = detectable_delta(n_items, rate, args.target_power, args.alpha)
    print(f"\nsmallest difference detected at {args.target_power:.0%} power: "
          f"{detectable * 100:.1f} pp", flush=True)

    print("\nitems needed at this discordance rate:", flush=True)
    by_target: dict[str, dict[str, int | None]] = {}
    for target in (0.8, 0.9):
        row: dict[str, int | None] = {}
        for delta in deltas:
            need = items_for_power(rate, delta, target, args.alpha)
            row[f"{delta:g}"] = need
            shown = f"{need}" if need is not None else ">100000"
            print(f"  {target:.0%} power at {delta * 100:5.1f} pp -> {shown} items",
                  flush=True)
        by_target[f"{target:g}"] = row

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "tier": args.tier or None,
            "items": n_items,
            "discordant": n_discordant,
            "discordance_rate": rate,
            "alpha": args.alpha,
            "target_power": args.target_power,
            "power_by_delta": by_delta,
            "detectable_delta": detectable,
            "items_by_power_and_delta": by_target,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"\nReport saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
