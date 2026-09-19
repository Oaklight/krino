"""Ordering bias and surface-form sensitivity metrics."""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable


def ordering_bias(
    scorer: Callable[[str, str, dict[str, str]], dict[str, float]],
    cases: list[dict[str, Any]],
    n_permutations: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure ordering bias by running all/sampled permutations of option order.

    Args:
        scorer: function(state, instructions, criteria) -> {option: probability}
        cases: list of {"state": ..., "instructions": ..., "options": {key: desc}}
        n_permutations: if set, sample this many permutations per case
        seed: random seed for permutation sampling
    """
    rng = random.Random(seed)
    all_records = []

    for ci, case in enumerate(cases):
        keys = list(case["options"].keys())
        n_opts = len(keys)
        perms = list(itertools.permutations(keys))
        if n_permutations and n_permutations < len(perms):
            perms = rng.sample(perms, n_permutations)

        for perm in perms:
            ordered = {k: case["options"][k] for k in perm}
            probs = scorer(case["state"], case["instructions"], ordered)
            all_records.append({
                "case_idx": ci,
                "option_order": list(perm),
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
            })

    # Position bias analysis
    if not all_records:
        return {"records": 0}

    max_opts = max(len(r["option_order"]) for r in all_records)
    position_means = {}
    for pos in range(max_opts):
        probs_at_pos = []
        for r in all_records:
            if pos < len(r["option_order"]):
                opt = r["option_order"][pos]
                probs_at_pos.append(r["probabilities"].get(opt, 0))
        if probs_at_pos:
            mean = sum(probs_at_pos) / len(probs_at_pos)
            expected = 1.0 / max_opts
            position_means[pos] = {"mean": mean, "delta": mean - expected, "n": len(probs_at_pos)}

    # Flip rate
    total_cases = len(cases)
    flip_count = 0
    swings = []
    for ci in range(total_cases):
        case_records = [r for r in all_records if r["case_idx"] == ci]
        if not case_records:
            continue
        choices = [r["choice"] for r in case_records]
        if len(set(choices)) > 1:
            flip_count += 1
        opts = list(case_records[0]["probabilities"].keys())
        for opt in opts:
            probs = [r["probabilities"].get(opt, 0) for r in case_records]
            if probs:
                swings.append(max(probs) - min(probs))

    return {
        "records": len(all_records),
        "position_bias": position_means,
        "flip_rate": flip_count / total_cases if total_cases > 0 else 0,
        "mean_swing": sum(swings) / len(swings) if swings else 0,
        "max_swing": max(swings) if swings else 0,
    }


def surface_form_sensitivity(
    scorer: Callable[[str, str, dict[str, str]], dict[str, float]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure how much option wording affects probability.

    Each item should have:
        state: str
        instructions: str
        variants: list of (variant_name, option_text, length, grammaticality, commonness)
        label_keys: list of opaque label keys to use
    """
    lengths, probabilities = [], []
    grammaticalities, commonnesses = [], []

    for item in items:
        criteria = {k: v[1] for k, v in zip(item["label_keys"], item["variants"])}
        probs = scorer(item["state"], item["instructions"], criteria)
        for key, variant in zip(item["label_keys"], item["variants"]):
            p = probs.get(key, 0)
            probabilities.append(p)
            lengths.append(variant[2])
            grammaticalities.append(variant[3])
            commonnesses.append(variant[4])

    if not probabilities:
        return {"n": 0}

    def pearson(xs: list[float], ys: list[float]) -> float | None:
        n = len(xs)
        if n < 3:
            return None
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        if sx < 1e-12 or sy < 1e-12:
            return None
        return cov / (sx * sy)

    r_length = pearson(lengths, probabilities)
    return {
        "n": len(probabilities),
        "length_correlation": r_length,
        "length_r_squared": r_length ** 2 if r_length is not None else None,
        "grammaticality_correlation": pearson(grammaticalities, probabilities),
        "commonness_correlation": pearson(commonnesses, probabilities),
    }
