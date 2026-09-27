"""LLM-based soft-label annotation for synthetic data items.

Uses any OpenAI-compatible chat model to generate probability distributions
for typed decision questions. Supports noul, choice, and score question types.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMLabelResult:
    item_id: str
    question_type: str
    teacher_probs: dict[str, float]
    teacher_label: Any
    model: str


_SYSTEM_CONTEXT = (
    "You are generating training labels for a faithful decision model — "
    "a classifier that outputs calibrated probability distributions over "
    "structured options. Your probability estimates will be used as soft "
    "teacher labels for knowledge distillation. Accuracy and calibration "
    "both matter: assign probabilities that reflect the true likelihood "
    "of each option being correct given the evidence."
)


def _build_noul_prompt(state: str, instructions: str) -> str:
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Task: binary (true/false) decision.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Estimate the probability that the correct answer is true. Express "
        f"genuine uncertainty — values like 0.75 or 0.3 are valid when the "
        f"evidence is ambiguous. Reserve extremes (>0.95 or <0.05) for cases "
        f"where the evidence is truly decisive.\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"probability_true": <float 0.0-1.0>}}'
    )


def _build_choice_prompt(state: str, instructions: str, criteria: dict[str, str]) -> str:
    options = "\n".join(f"  - {k}: {v}" for k, v in criteria.items())
    keys = list(criteria.keys())
    prob_fields = ", ".join(f'"{k}": <float>' for k in keys)
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Task: multi-option classification.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Options:\n{options}\n\n"
        f"Distribute probability across these MUTUALLY EXCLUSIVE options. "
        f"The probabilities MUST sum to exactly 1.0 — this is a categorical "
        f"distribution, NOT independent confidence scores. Assign each option "
        f"a share of the total probability reflecting how likely it is the "
        f"correct answer.\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"choice": "<best_option_key>", "probabilities": {{{prob_fields}}}}}'
    )


def _build_score_prompt(state: str, instructions: str, criteria: list[str]) -> str:
    levels = "\n".join(f"  {i}: {c}" for i, c in enumerate(criteria))
    prob_fields = ", ".join(f'"{i}": <float>' for i in range(len(criteria)))
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Task: ordinal rating.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Score levels (0-indexed):\n{levels}\n\n"
        f"Distribute probability across these MUTUALLY EXCLUSIVE score levels. "
        f"The probabilities MUST sum to exactly 1.0 — this is a categorical "
        f"distribution over ordinal levels. Adjacent levels may share "
        f"probability when the rating is borderline.\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"score": <float 0.0-{len(criteria)-1}.0>, "probabilities": {{{prob_fields}}}}}'
    )


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _parse_json_content(content: str) -> dict | None:
    text = _THINK_RE.sub("", content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_noul_response(content: str) -> dict[str, float] | None:
    parsed = _parse_json_content(content)
    if not parsed:
        return None
    try:
        p = float(parsed["probability_true"])
        return {"true": p, "false": 1.0 - p}
    except (KeyError, ValueError, TypeError):
        return None


def _parse_choice_response(content: str, valid_keys: list[str]) -> tuple[str | None, dict[str, float] | None]:
    parsed = _parse_json_content(content)
    if not parsed:
        return None, None
    try:
        choice = parsed.get("choice", "")
        probs = parsed.get("probabilities", {})
        if choice not in valid_keys:
            for k in valid_keys:
                if k.lower() == choice.lower():
                    choice = k
                    break
        probs = {k: float(v) for k, v in probs.items() if k in valid_keys}
        return choice, probs if probs else None
    except (ValueError, TypeError):
        return None, None


def _parse_score_response(content: str, n_levels: int) -> tuple[float | None, dict[str, float] | None]:
    parsed = _parse_json_content(content)
    if not parsed:
        return None, None
    try:
        score = float(parsed.get("score", -1))
        probs = {str(k): float(v) for k, v in parsed.get("probabilities", {}).items()}
        return score, probs if probs else None
    except (ValueError, TypeError):
        return None, None


async def label_items_llm(
    client: Any,
    items: list[dict[str, Any]],
    model: str,
    base_url: str,
    max_concurrent: int = 10,
    max_retries: int = 3,
) -> list[LLMLabelResult]:
    """Label items using an LLM via chat completions."""

    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[LLMLabelResult | None] = [None] * len(items)

    async def _label_one(idx: int, item: dict) -> None:
        q = item["question"]
        q_type = q["type"]
        state = item["state"]

        # Augmented noul items have multi-option alternatives — label as choice
        augmented_options = q.get("augmented_options") if q_type == "noul" else None
        effective_type = "choice" if augmented_options else q_type

        if effective_type == "noul":
            prompt = _build_noul_prompt(state, q["instructions"])
        elif effective_type == "choice":
            criteria = augmented_options if augmented_options else q["criteria"]
            prompt = _build_choice_prompt(state, q["instructions"], criteria)
        elif effective_type == "score":
            prompt = _build_score_prompt(state, q["instructions"], q["criteria"])
        else:
            return

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        }
        # Reasoning models (o1, o3, gpt-5.6-*) don't support temperature=0
        is_reasoning = any(tag in model.lower() for tag in ("o1", "o3", "o4", "5.6", "5.5"))
        if is_reasoning:
            payload["reasoning_effort"] = "high"
        else:
            payload["temperature"] = 0
        # Qwen3.8 models: disable thinking mode for clean JSON output
        if "qwen" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    resp = await client.post(
                        f"{base_url}/v1/chat/completions", json=payload
                    )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    logger.warning(
                        "LLM label error %d for %s (attempt %d)",
                        resp.status_code, item["id"], attempt + 1,
                    )
                    await asyncio.sleep(1)
                    continue

                content = resp.json()["choices"][0]["message"]["content"]

                if effective_type == "noul":
                    probs = _parse_noul_response(content)
                    if probs:
                        results[idx] = LLMLabelResult(
                            item_id=item["id"],
                            question_type=q_type,
                            teacher_probs=probs,
                            teacher_label=probs["true"] >= 0.5,
                            model=model,
                        )
                        return

                elif effective_type == "choice":
                    criteria = augmented_options if augmented_options else q["criteria"]
                    choice, probs = _parse_choice_response(content, list(criteria.keys()))
                    if probs:
                        results[idx] = LLMLabelResult(
                            item_id=item["id"],
                            question_type=q_type,
                            teacher_probs=probs,
                            teacher_label=choice,
                            model=model,
                        )
                        return

                elif effective_type == "score":
                    score, probs = _parse_score_response(content, len(q["criteria"]))
                    if probs:
                        results[idx] = LLMLabelResult(
                            item_id=item["id"],
                            question_type=q_type,
                            teacher_probs=probs,
                            teacher_label=score,
                            model=model,
                        )
                        return

            except Exception as exc:
                logger.warning("LLM label failed for %s (attempt %d): %s", item["id"], attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)

    tasks = [_label_one(i, item) for i, item in enumerate(items)]

    done = 0
    total = len(tasks)
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 100 == 0 or done == total:
            logger.info("  LLM labeling: %d/%d done", done, total)

    valid = [r for r in results if r is not None]
    logger.info("  LLM labeling complete: %d/%d items labeled", len(valid), len(items))
    return valid
