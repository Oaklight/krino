#!/usr/bin/env python3
"""Paired McNemar tests between JevBench prediction dumps.

The hard tier is 111 items. An unpaired standard error is about 4.7 points, which
is large enough that a 4-point difference between two models can be one model
bettering the other on a handful of the same items -- or on different items, which
is a weaker claim. This compares the dumps item by item.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def load(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    return {row["id"]: bool(row["correct"]) for row in payload}


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
    data = {name: load(Path(path)) for name, path in runs}
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
            wins, losses, p = mcnemar(data[first], data[second], ids)
            print(
                f"{first} vs {second}: +{wins} / -{losses} discordant, "
                f"McNemar p={p:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
