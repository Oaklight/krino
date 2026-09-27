"""Normalize probability distributions in teacher_probs.

Ensures all probability distributions sum to 1.0. Handles three failure modes
observed in LLM-generated teacher labels:
1. All-high (~3.96): LLM assigned ~0.99 to every option without softmax
2. Double (~1.98): LLM was confident in 2 options, both got ~0.99
3. Truncated (<0.5): parsing lost some options, remaining probs are too low
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOLERANCE = 0.005


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Normalize a probability distribution to sum to 1.0.

    Args:
        probs: Option key -> probability mapping.

    Returns:
        Normalized copy if |sum - 1| > TOLERANCE, otherwise the original dict.
    """
    if not probs:
        return probs

    total = sum(probs.values())
    if total <= 0:
        return probs

    if abs(total - 1.0) <= TOLERANCE:
        return probs

    return {k: v / total for k, v in probs.items()}


def normalize_teacher_probs(
    teacher_probs: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Normalize all probability distributions in a teacher_probs dict.

    Args:
        teacher_probs: {teacher_name: {option: prob, ...}, ...}

    Returns:
        (normalized_dict, correction_count)
    """
    corrected = 0
    result = {}
    for teacher, probs in teacher_probs.items():
        if not isinstance(probs, dict):
            result[teacher] = probs
            continue
        normalized = normalize_probs(probs)
        if normalized is not probs:
            corrected += 1
        result[teacher] = normalized
    return result, corrected


def normalize_items_in_place(
    items: list[dict[str, Any]],
) -> int:
    """Normalize teacher_probs across a list of items, modifying in place.

    Returns:
        Total number of distributions corrected.
    """
    total_corrected = 0
    for item in items:
        tp = item.get("teacher_probs")
        if not tp or not isinstance(tp, dict):
            continue
        normalized, corrected = normalize_teacher_probs(tp)
        if corrected:
            item["teacher_probs"] = normalized
            total_corrected += corrected
    return total_corrected


def normalize_label_file(
    labels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Normalize teacher_probs in a list of label dicts (from *_labels.jsonl).

    Returns:
        (normalized_labels, correction_count)
    """
    total_corrected = 0
    for lbl in labels:
        tp = lbl.get("teacher_probs")
        if not tp or not isinstance(tp, dict):
            continue
        normalized, corrected = normalize_teacher_probs(tp)
        if corrected:
            lbl["teacher_probs"] = normalized
            total_corrected += corrected
    return labels, total_corrected
