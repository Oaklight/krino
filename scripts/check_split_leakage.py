#!/usr/bin/env python3
"""Measure state leakage across the synthetic train/eval split.

Rebuilds the near-duplicate clusters the loader uses and reports, per domain,
how many cross-split family pairs still share a state at or above the Jaccard
threshold, verified exactly (the cluster decision itself is MinHash-approximate).
Also reports how many families the cluster-aware split moved relative to the
family-level split it replaces (#120).

Usage:
    python scripts/check_split_leakage.py
    python scripts/check_split_leakage.py --threshold 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.lsh import LSHIndex, MinHash, char_ngrams  # noqa: E402
from data.pipeline import DATA_DIR, _synthetic_family_split  # noqa: E402
from data.synthetic import _load_jsonl  # noqa: E402
from data.synthetic_split import (  # noqa: E402
    DEFAULT_BANDS,
    DEFAULT_NUM_PERM,
    assign_splits,
    cluster_families,
    family_state_texts,
)
from data.synthetic_templates import DOMAIN_TEMPLATES  # noqa: E402


def exact_jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard over precomputed character 5-gram sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cross_split_leaks(
    entries: list[tuple[str, str]],
    threshold: float,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
) -> int:
    """Count cross-split state pairs whose exact Jaccard reaches ``threshold``.

    Candidates come from a fresh LSH index so the check stays proportional to
    the number of near-duplicates rather than quadratic in states. Signatures
    are built from the n-gram sets directly, so they hash the same features the
    clustering saw.

    Args:
        entries: ``(state text, split)`` per state text.
        threshold: Jaccard at or above which a pair counts as leaking.
        num_perm: MinHash permutations for candidate generation.
        bands: LSH bands for candidate generation.

    Returns:
        Number of leaking pairs.
    """
    lsh = LSHIndex(num_perm=num_perm, bands=bands)
    signatures: list[tuple[int, ...]] = []
    leaks = 0
    for text, split in entries:
        minhash = MinHash(num_perm=num_perm, seed=lsh.seed)
        minhash.update(char_ngrams(text))
        signature = minhash.signature
        for candidate in lsh.query(signature):
            other_text, other_split = entries[candidate]
            if other_split == split:
                continue
            # Re-extract n-grams only for pairs that survive banding, so the
            # corpus is never held in memory as n-gram sets all at once.
            if exact_jaccard(
                char_ngrams(text), char_ngrams(other_text)
            ) >= threshold:
                leaks += 1
        lsh.insert(len(signatures), signature)
        signatures.append(signature)
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    synth_dir = DATA_DIR / "synthetic"
    leaks_total = 0
    moved_total = 0
    families_total = 0

    for domain in DOMAIN_TEMPLATES:
        families_path = synth_dir / f"{domain}_families.jsonl"
        variants_path = synth_dir / f"{domain}_variants.jsonl"
        if not families_path.exists():
            continue

        families = _load_jsonl(families_path)
        variants = _load_jsonl(variants_path) if variants_path.exists() else []
        family_idxs = [fam.get("_family_idx", i) for i, fam in enumerate(families)]
        if family_idxs != list(range(len(families))):
            print(
                f"WARNING {domain}: _family_idx is not positional; "
                "the moved-split count is approximate",
                file=sys.stderr,
            )

        states = [
            family_state_texts(fam, variants[i] if i < len(variants) else None)
            for i, fam in enumerate(families)
        ]
        clusters = cluster_families(states, threshold=args.threshold)
        splits = assign_splits(domain, clusters, args.eval_ratio)

        by_cluster: dict[int, set[str]] = {}
        for label, split in zip(clusters, splits):
            by_cluster.setdefault(label, set()).add(split)
        straddling = sum(1 for s in by_cluster.values() if len(s) > 1)

        old_splits = [
            _synthetic_family_split(domain, idx, args.eval_ratio)
            for idx in family_idxs
        ]
        moved = sum(1 for o, n in zip(old_splits, splits) if o != n)

        entries = [
            (text, splits[i])
            for i, fam_states in enumerate(states)
            for text in fam_states
        ]
        leaks = cross_split_leaks(entries, args.threshold)

        leaks_total += leaks + straddling
        moved_total += moved
        families_total += len(families)
        if leaks or straddling or moved:
            print(
                f"{domain:<26} families={len(families):>4} "
                f"moved={moved:>3} clusters={len(by_cluster):>4} "
                f"straddling={straddling} leaking_pairs={leaks}"
            )

    print(
        f"\n{families_total} families, {moved_total} moved by clustering, "
        f"{leaks_total} residual cross-split pairs at jaccard >= {args.threshold}"
    )
    return 0 if leaks_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
