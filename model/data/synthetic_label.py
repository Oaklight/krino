"""Jev API soft-label annotation for synthetic data items."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONFIDENCE_BUCKETS = {
    "high": (0.9, 1.01),
    "medium": (0.6, 0.9),
    "uncertain": (0.0, 0.6),
}


@dataclass
class LabelResult:
    question_id: str
    question_type: str
    teacher_probs: dict[str, float]
    jev_confidence: float
    bucket: str
    latency_s: float


def _bucket_confidence(confidence: float) -> str:
    for bucket, (lo, hi) in CONFIDENCE_BUCKETS.items():
        if lo <= confidence < hi:
            return bucket
    return "uncertain"


def _extract_probs(answer: dict[str, Any], question_type: str) -> tuple[dict[str, float], float]:
    """Extract probability distribution from a Jev API answer."""
    if question_type == "noul":
        p = answer["noul"]
        return {"true": p, "false": 1.0 - p}, max(p, 1.0 - p)

    if question_type == "choice":
        probs = answer.get("probabilities", {})
        confidence = answer.get("confidence", max(probs.values()) if probs else 0.0)
        return probs, confidence

    if question_type == "score":
        probs = answer.get("probabilities", {})
        confidence = answer.get("confidence", max(probs.values()) if probs else 0.0)
        return probs, confidence

    raise ValueError(f"Unknown question type: {question_type}")


def label_items_sync(
    items: list[dict[str, Any]],
    jev_client: Any,
    batch_size: int = 5,
    delay_between_batches: float = 0.1,
) -> list[LabelResult]:
    """Label items with Jev API soft labels (synchronous, batched).

    Groups items by state to minimize API calls — Jev supports multiple
    questions per request.
    """
    by_state: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_state.setdefault(item["state"], []).append(item)

    results: list[LabelResult] = []
    states = list(by_state.items())

    for batch_start in range(0, len(states), batch_size):
        batch = states[batch_start:batch_start + batch_size]

        for state, state_items in batch:
            questions = {}
            item_map = {}
            for item in state_items:
                qid = item["id"]
                q = dict(item["question"])
                q.pop("instructions_original", None)
                questions[qid] = q
                item_map[qid] = item

            t0 = time.monotonic()
            try:
                resp = jev_client.ask(state, questions)
                latency = time.monotonic() - t0
            except Exception as exc:
                latency = time.monotonic() - t0
                logger.warning("Jev API error for state (%.0fs): %s", latency, exc)
                for qid, item in item_map.items():
                    results.append(LabelResult(
                        question_id=qid,
                        question_type=item["question"]["type"],
                        teacher_probs={},
                        jev_confidence=0.0,
                        bucket="uncertain",
                        latency_s=latency,
                    ))
                continue

            answers = resp.get("answers", {})
            for qid, answer in answers.items():
                item = item_map.get(qid)
                if not item:
                    continue
                q_type = item["question"]["type"]
                try:
                    probs, confidence = _extract_probs(answer, q_type)
                    bucket = _bucket_confidence(confidence)
                except (KeyError, TypeError) as exc:
                    logger.warning("Failed to extract probs for %s: %s", qid, exc)
                    probs, confidence, bucket = {}, 0.0, "uncertain"

                results.append(LabelResult(
                    question_id=qid,
                    question_type=q_type,
                    teacher_probs=probs,
                    jev_confidence=confidence,
                    bucket=bucket,
                    latency_s=latency,
                ))

        if batch_start + batch_size < len(states):
            time.sleep(delay_between_batches)

    return results


def print_bucket_report(results: list[LabelResult]) -> None:
    """Print a summary of confidence bucket distribution."""
    counts: dict[str, int] = {"high": 0, "medium": 0, "uncertain": 0}
    for r in results:
        counts[r.bucket] = counts.get(r.bucket, 0) + 1
    total = len(results)
    print(f"\n  Jev confidence buckets ({total} items):")
    for bucket in ["high", "medium", "uncertain"]:
        lo, hi = CONFIDENCE_BUCKETS[bucket]
        count = counts.get(bucket, 0)
        pct = 100 * count / total if total else 0
        print(f"    {bucket:<12} [{lo:.1f}, {hi:.1f}): {count:>6,} ({pct:5.1f}%)")
