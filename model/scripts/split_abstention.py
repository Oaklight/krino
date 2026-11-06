"""Split abstention sets into train and held-out eval, with paired positives.

The builder produces constructed negatives (``answerable=False``).  This script
splits them by *base id* so a base item cannot appear in both splits, then builds
the eval mix the risk-coverage metric needs:

- negatives: the held-out abstention items;
- positives: the original corpus item for each held-out base id, answerable and
  carrying its gold label.

The split is grouped by base id, so a base item that produced two reasons stays
on one side.  ``--eval-base-ids-out`` lists the held-out base ids; pass those to
the answerable-head training as exclusions so the head never trains on an eval
item in answerable form.

Usage:
    python model/scripts/split_abstention.py \
        --abstention data/benchmarks/abstention_train_20k.jsonl \
        --corpus data/benchmarks/residual_train_20000_jev.jsonl \
        --out-dir data/benchmarks/abstention_20k \
        --eval-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import TypedQuestion  # noqa: E402
from data.pipeline import load_jsonl, save_jsonl  # noqa: E402

#: Derived-id suffixes written by the abstention builders.  ``-abstain-`` is
#: build_abstention_sets.py; ``-unans-`` is build_unanswerable_pairs.py.
SUFFIXES = ("-abstain-", "-unans-")


def base_id(item: TypedQuestion) -> str:
    """Derived id -> the id of the item it was built from."""
    for suffix in SUFFIXES:
        head, sep, _tail = item.id.partition(suffix)
        if sep:
            return head
    return item.id


def split(
    abstention: list[TypedQuestion],
    corpus: list[TypedQuestion],
    eval_fraction: float,
    seed: int,
    eval_overrides: set[str] | None = None,
) -> dict[str, list[TypedQuestion]]:
    """Return train, eval negatives and paired eval positives.

    ``eval_overrides`` forces the given base ids into the eval split.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")

    by_base: dict[str, list[TypedQuestion]] = defaultdict(list)
    for item in abstention:
        by_base[base_id(item)].append(item)

    bases = sorted(by_base)
    rng = random.Random(seed)
    rng.shuffle(bases)
    target = max(1, int(round(len(bases) * eval_fraction)))
    # Overrides may name ids the abstention set does not contain; drop them so
    # the exclusion list matches what the split actually holds out.
    eval_bases = set(eval_overrides or ()) & set(by_base)
    target = max(target, len(eval_bases))
    for base in bases:
        if len(eval_bases) >= target:
            break
        eval_bases.add(base)

    train: list[TypedQuestion] = []
    eval_negatives: list[TypedQuestion] = []
    for base in sorted(by_base):
        destination = eval_negatives if base in eval_bases else train
        for item in by_base[base]:
            destination.append(replace(item, split="test" if base in eval_bases else "train"))

    corpus_by_id: dict[str, TypedQuestion] = {}
    for item in corpus:
        corpus_by_id.setdefault(item.id, item)

    eval_positives: list[TypedQuestion] = []
    for base in sorted(eval_bases):
        item = corpus_by_id.get(base)
        if item is None:
            continue
        eval_positives.append(replace(item, split="test", answerable=True, reason=None))

    return {
        "train": train,
        "eval_negatives": eval_negatives,
        "eval_positives": eval_positives,
        "eval_bases": sorted(eval_bases),
    }


def _stats(parts: dict, source: str) -> dict:
    train = parts["train"]
    negatives = parts["eval_negatives"]
    positives = parts["eval_positives"]
    train_bases = {base_id(item) for item in train}
    eval_bases = set(parts["eval_bases"])
    missing_positive = sorted(eval_bases - {item.id for item in positives})
    return {
        "abstention": source,
        "train": len(train),
        "train_by_reason": dict(Counter(item.reason for item in train)),
        "eval_negatives": len(negatives),
        "eval_negatives_by_reason": dict(Counter(item.reason for item in negatives)),
        "eval_positives": len(positives),
        "eval_mix": len(negatives) + len(positives),
        "eval_base_ids": len(eval_bases),
        "base_id_overlap": len(train_bases & eval_bases),
        "eval_bases_without_positive": missing_positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstention", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-base-ids", type=Path, default=None,
                        help="force these base ids into eval (one per line)")
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    abstention = load_jsonl(args.abstention)
    corpus = load_jsonl(args.corpus)
    if not abstention:
        raise SystemExit(f"no items in {args.abstention}")

    overrides = None
    if args.eval_base_ids:
        overrides = {
            line.strip() for line in args.eval_base_ids.read_text().splitlines()
            if line.strip()
        }

    parts = split(abstention, corpus, args.eval_fraction, args.seed, overrides)
    stats = _stats(parts, str(args.abstention))
    if stats["base_id_overlap"]:
        raise SystemExit(f"base id leaked across splits: {stats['base_id_overlap']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(parts["train"], args.out_dir / "train.jsonl")
    save_jsonl(parts["eval_negatives"], args.out_dir / "eval_negatives.jsonl")
    save_jsonl(parts["eval_positives"], args.out_dir / "eval_positives.jsonl")
    eval_mix = parts["eval_negatives"] + parts["eval_positives"]
    save_jsonl(eval_mix, args.out_dir / "eval_mix.jsonl")
    (args.out_dir / "eval_base_ids.txt").write_text(
        "".join(f"{base}\n" for base in parts["eval_bases"])
    )

    stats_path = args.stats or (args.out_dir / "split.stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote split to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
