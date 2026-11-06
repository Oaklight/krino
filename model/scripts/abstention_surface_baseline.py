"""Surface-cue baseline for the paired unanswerable corpus.

A model could learn "unanswerable" from a surface difference between a variant
and its answerable original (length, token count, punctuation, source mix)
instead of from the missing or corrupted evidence.  This script measures that
risk with features that carry no semantics at all.

Two linear baselines, both with a ridge classifier over surface features only:

``pair_discrimination``
    Two rows per pair, the original and one variant.  Can the classifier tell
    which member is the variant from surface form alone?  Pairs are split
    train/test by base id, so neither member of a test pair is seen in
    training.  AUC 0.5 is chance; a higher AUC is a surface cue.
``negative_vs_positive``
    The paired originals against their variants, split by base id.

Features: log state length, log token count, log sentence count, mean token
length, digit fraction, uppercase fraction, punctuation fraction, source
one-hot, question type one-hot, and option count.  Donor identity is not a
feature: it does not exist at inference time.  The donor source is the same
source by construction, so source one-hot is the proxy the peer asked for.

The classifier is a ridge regression on +1/-1 labels, implemented in plain
Python so the check runs without numpy or torch.  The null band comes from
permuting the test labels.

Usage:
    python model/scripts/abstention_surface_baseline.py \
        --pairs data/benchmarks/unanswerable_30k/pairs.jsonl \
        --out results/abstention/unanswerable_30k.surface.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SENTENCE = re.compile(r"[.!?]\s+")


def _read_pairs(path: Path) -> tuple[dict[str, dict], list[tuple[str, str, dict]]]:
    """Return originals by base id and ``(base, reason, variant)`` rows."""
    originals: dict[str, dict] = {}
    variants: list[tuple[str, str, dict]] = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answerable") is True:
            originals[row["id"]] = row
            continue
        reason = row.get("reason")
        if not reason:
            continue
        base = row["id"].split("-unans-", 1)[0]
        variants.append((base, str(reason), row))
    return originals, variants


def _numeric_features(row: dict) -> list[float]:
    state = row.get("state") or ""
    tokens = state.split()
    length = max(1, len(state))
    token_count = max(1, len(tokens))
    sentences = len(SENTENCE.split(state)) if state.strip() else 0
    question = row.get("question") or {}
    criteria = question.get("criteria")
    n_options = len(criteria) if isinstance(criteria, dict) else 0
    return [
        math.log1p(len(state)),
        math.log1p(token_count),
        math.log1p(sentences),
        sum(len(token) for token in tokens) / token_count,
        sum(character.isdigit() for character in state) / length,
        sum(character.isupper() for character in state) / length,
        sum(
            (not character.isalnum()) and (not character.isspace())
            for character in state
        )
        / length,
        float(n_options),
    ]


NUMERIC_NAMES = [
    "log_chars",
    "log_tokens",
    "log_sentences",
    "mean_token_len",
    "digit_fraction",
    "upper_fraction",
    "punctuation_fraction",
    "n_options",
]


def _feature_names(rows: list[dict]) -> list[str]:
    sources = sorted({str(row.get("source")) for row in rows})
    types = sorted({str((row.get("question") or {}).get("type")) for row in rows})
    return (
        NUMERIC_NAMES
        + [f"source={source}" for source in sources]
        + [f"type={qtype}" for qtype in types]
    )


def _features(
    row: dict, sources: list[str], types: list[str]
) -> list[float]:
    source = str(row.get("source"))
    qtype = str((row.get("question") or {}).get("type"))
    return (
        _numeric_features(row)
        + [1.0 if source == name else 0.0 for name in sources]
        + [1.0 if qtype == name else 0.0 for name in types]
    )


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    size = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            continue
        rows[column], rows[pivot] = rows[pivot], rows[column]
        lead = rows[column][column]
        for index in range(column, size + 1):
            rows[column][index] /= lead
        for row in range(size):
            if row == column:
                continue
            factor = rows[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                rows[row][index] -= factor * rows[column][index]
    return [rows[index][size] for index in range(size)]


def _fit(
    features: list[list[float]], labels: list[int], ridge: float
) -> tuple[list[float], list[float], list[float]]:
    """Fit a ridge classifier and return weights, means, and scales."""
    width = len(features[0])
    means = [
        sum(row[column] for row in features) / len(features)
        for column in range(width)
    ]
    scales = []
    for column in range(width):
        variance = sum(
            (row[column] - means[column]) ** 2 for row in features
        ) / len(features)
        scales.append(math.sqrt(variance) or 1.0)
    standardized = [
        [(row[column] - means[column]) / scales[column] for column in range(width)]
        + [1.0]
        for row in features
    ]
    width += 1
    matrix = [[0.0] * width for _ in range(width)]
    vector = [0.0] * width
    for row, label in zip(standardized, labels):
        active = [index for index, value in enumerate(row) if value != 0.0]
        for left in active:
            vector[left] += row[left] * label
            for right in active:
                matrix[left][right] += row[left] * row[right]
    for index in range(width - 1):
        matrix[index][index] += ridge
    return _solve(matrix, vector), means, scales


def _predict(
    weights: list[float],
    means: list[float],
    scales: list[float],
    features: list[list[float]],
) -> list[float]:
    scores = []
    for row in features:
        total = weights[-1]
        for column, value in enumerate(row):
            total += weights[column] * (value - means[column]) / scales[column]
        scores.append(total)
    return scores


def _auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _null_band(
    scores: list[float], labels: list[int], draws: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    shuffled = list(labels)
    band = []
    for _ in range(draws):
        rng.shuffle(shuffled)
        band.append(_auc(scores, shuffled))
    band.sort()
    positives = sum(labels)
    negatives = len(labels) - positives
    null_sd = math.sqrt(
        (positives + negatives + 1) / (12.0 * positives * negatives)
    )
    return {
        "null_p05": band[int(0.05 * (len(band) - 1))],
        "null_p50": band[len(band) // 2],
        "null_p95": band[int(0.95 * (len(band) - 1))],
        "null_sd": null_sd,
    }


def _evaluate(
    rows: list[tuple[dict, int]],
    seed: int,
    train_fraction: float,
    ridge: float,
    draws: int,
) -> dict:
    """Fit and score one surface-only experiment."""
    if not rows:
        return {"n": 0}
    bases = sorted({str(row.get("_base")) for row, _label in rows})
    rng = random.Random(seed)
    rng.shuffle(bases)
    split = max(1, int(round(len(bases) * train_fraction)))
    train_bases = set(bases[:split])
    train = [
        (row, label)
        for row, label in rows
        if str(row.get("_base")) in train_bases
    ]
    test = [
        (row, label)
        for row, label in rows
        if str(row.get("_base")) not in train_bases
    ]
    if not train or not test:
        return {"n": 0}
    sources = sorted({str(row.get("source")) for row, _label in rows})
    types = sorted(
        {str((row.get("question") or {}).get("type")) for row, _label in rows}
    )
    train_features = [_features(row, sources, types) for row, _label in train]
    train_labels = [label for _row, label in train]
    test_features = [_features(row, sources, types) for row, _label in test]
    test_labels = [label for _row, label in test]
    weights, means, scales = _fit(train_features, train_labels, ridge)
    scores = _predict(weights, means, scales, test_features)
    auc = _auc(scores, test_labels)
    result = {
        "pairs": len({str(row.get("_base")) for row, _label in rows}),
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "auc": auc,
    }
    result.update(_null_band(scores, test_labels, draws, seed))
    if result["null_sd"]:
        result["z"] = (auc - 0.5) / result["null_sd"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args(argv)

    originals, variants = _read_pairs(args.pairs)
    if not originals or not variants:
        raise SystemExit("no paired rows found")

    paired = [
        (base, reason, row)
        for base, reason, row in variants
        if base in originals
    ]
    all_rows = list(originals.values()) + [row for _base, _reason, row in paired]
    feature_names = _feature_names(all_rows)

    pair_rows: list[tuple[dict, int]] = []
    by_reason: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for base, reason, variant in paired:
        original = originals[base]
        for row, label in ((original, 0), (variant, 1)):
            row = dict(row)
            row["_base"] = base
            pair_rows.append((row, label))
            by_reason[reason].append((row, label))

    pair_result = _evaluate(
        pair_rows, args.seed, args.train_fraction, args.ridge, args.permutations
    )
    pair_by_reason = {
        reason: _evaluate(
            rows, args.seed, args.train_fraction, args.ridge, args.permutations
        )
        for reason, rows in sorted(by_reason.items())
    }

    negative_rows: list[tuple[dict, int]] = []
    negative_by_reason: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for base, reason, variant in paired:
        original = dict(originals[base])
        original["_base"] = base
        variant = dict(variant)
        variant["_base"] = base
        negative_rows.extend([(original, 0), (variant, 1)])
        negative_by_reason[reason].extend([(original, 0), (variant, 1)])

    negative_result = _evaluate(
        negative_rows, args.seed + 1, args.train_fraction, args.ridge, args.permutations
    )
    negative_by_reason = {
        reason: _evaluate(
            rows, args.seed + 1, args.train_fraction, args.ridge, args.permutations
        )
        for reason, rows in sorted(negative_by_reason.items())
    }

    result = {
        "pairs": str(args.pairs),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "features": feature_names,
        "pair_discrimination": {
            "overall": pair_result,
            "by_reason": pair_by_reason,
        },
        "negative_vs_positive": {
            "overall": negative_result,
            "by_reason": negative_by_reason,
        },
        "reading": (
            "AUC near 0.5 and inside the permutation band means surface form "
            "does not separate the pair.  An AUC above the band is a cue the "
            "answerable head could learn instead of the evidence."
        ),
    }
    payload = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
