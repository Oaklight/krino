#!/usr/bin/env python3
"""Generate deterministic variants of an eval set for robustness probing.

Three transformations, each aimed at a specific failure mode that a single
accuracy number cannot see:

* ``neutral`` — replace every question's instructions with one fixed template.
  Separates "can the model decide from the state" from "did it memorise our
  instruction phrasings".
* ``reversed`` — reverse the option/level order and remap the label. A model
  that scores by position rather than content loses accuracy here; a correct
  one is unchanged.
* ``shuffled`` — permute options/levels with a fixed seed, remap labels.

Applied to a prepared JSONL (``data/benchmarks/<source>.jsonl``), the output is
another JSONL in the same format, so it can be evaluated by pointing an
eval-only config at it. No model code is involved.

Usage:
    python scripts/make_eval_variants.py data/benchmarks/jevbench.jsonl \
        --out-dir data/benchmarks/variants --variants neutral,reversed,shuffled
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

NEUTRAL_INSTRUCTION = "Answer the question using only the information in the text."


def reverse_choice(criteria: dict[str, str], label: str) -> tuple[dict[str, str], str]:
    """Reverse a choice option mapping; the label is unchanged because keys move."""
    return {k: criteria[k] for k in reversed(list(criteria))}, label


def reverse_score(criteria: list[str], label: float) -> tuple[list[str], float]:
    """Reverse an ordinal level list and mirror the expected-level label."""
    n_levels = len(criteria)
    return list(reversed(criteria)), (n_levels - 1) - label


def shuffle_choice(
    criteria: dict[str, str], label: str, rng: random.Random
) -> tuple[dict[str, str], str]:
    """Permute a choice option mapping; the label key is unchanged."""
    keys = list(criteria)
    rng.shuffle(keys)
    return {k: criteria[k] for k in keys}, label


def shuffle_score(
    criteria: list[str], label: float, rng: random.Random
) -> tuple[list[str], float]:
    """Permute ordinal levels and move the label to the level's new index."""
    order = list(range(len(criteria)))
    rng.shuffle(order)
    new_levels = [criteria[i] for i in order]
    # ``order[new_index] == old_index``, so the label's new index is its position.
    return new_levels, float(order.index(int(round(label))))


def transform_item(item: dict[str, Any], variant: str, rng: random.Random) -> dict[str, Any]:
    """Apply one variant to an item dict, returning a new dict.

    Args:
        item: A prepared ``TypedQuestion`` dict.
        variant: ``neutral``, ``reversed`` or ``shuffled``.
        rng: Seeded generator, used for ``shuffled``.

    Returns:
        A transformed copy. The id gets a suffix so the variant is traceable.

    Raises:
        ValueError: On an unknown variant or question type.
    """
    out = json.loads(json.dumps(item))  # cheap deep copy
    out["id"] = f"{item['id']}::{variant}"
    question = out["question"]
    q_type = question.get("type", "noul")

    if variant == "neutral":
        question["instructions"] = NEUTRAL_INSTRUCTION
        return out

    if variant == "reversed":
        if q_type == "choice":
            question["criteria"], out["label"] = reverse_choice(
                question["criteria"], out["label"]
            )
        elif q_type == "score":
            question["criteria"], out["label"] = reverse_score(
                question["criteria"], float(out["label"])
            )
        return out

    if variant == "shuffled":
        if q_type == "choice":
            question["criteria"], out["label"] = shuffle_choice(
                question["criteria"], out["label"], rng
            )
        elif q_type == "score":
            question["criteria"], out["label"] = shuffle_score(
                question["criteria"], float(out["label"]), rng
            )
        return out

    raise ValueError(f"unknown variant {variant!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval-set robustness variants")
    parser.add_argument("path", type=Path, help="Prepared JSONL to transform")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--variants", default="neutral,reversed,shuffled")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir or args.path.parent / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line]
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        rng = random.Random(args.seed)
        transformed = [transform_item(row, variant, rng) for row in rows]
        out_path = out_dir / f"{args.path.stem}_{variant}.jsonl"
        out_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transformed)
        )
        print(f"  {variant}: {len(transformed)} items -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
