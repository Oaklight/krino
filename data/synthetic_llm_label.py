"""LLM-based soft-label annotation for synthetic data items.

Uses any OpenAI-compatible chat model to generate probability distributions
for typed decision questions. Supports noul, choice, and score question types.
"""

from __future__ import annotations

import asyncio
import json
import logging
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


def _build_noul_prompt(state: str, instructions: str) -> str:
    return (
        f"You are evaluating whether a proposition is true or false.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"probability_true": <float 0.0-1.0>}}'
    )


def _build_choice_prompt(state: str, instructions: str, criteria: dict[str, str]) -> str:
    options = "\n".join(f"  - {k}: {v}" for k, v in criteria.items())
    keys = list(criteria.keys())
    prob_fields = ", ".join(f'"{k}": <float>' for k in keys)
    return (
        f"You are evaluating which option best applies.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Options:\n{options}\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"choice": "<option_key>", "probabilities": {{{prob_fields}}}}}\n'
        f"Probabilities must sum to 1.0."
    )


def _build_score_prompt(state: str, instructions: str, criteria: list[str]) -> str:
    levels = "\n".join(f"  {i}: {c}" for i, c in enumerate(criteria))
    prob_fields = ", ".join(f'"{i}": <float>' for i in range(len(criteria)))
    return (
        f"You are rating on an ordinal scale.\n\n"
        f"State:\n{state}\n\n"
        f"Question: {instructions}\n\n"
        f"Score levels (0-indexed):\n{levels}\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"score": <float 0.0-{len(criteria)-1}.0>, "probabilities": {{{prob_fields}}}}}\n'
        f"Probabilities must sum to 1.0."
    )


def _parse_json_content(content: str) -> dict | None:
    text = content.strip()
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

        if q_type == "noul":
            prompt = _build_noul_prompt(state, q["instructions"])
        elif q_type == "choice":
            prompt = _build_choice_prompt(state, q["instructions"], q["criteria"])
        elif q_type == "score":
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

                if q_type == "noul":
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

                elif q_type == "choice":
                    choice, probs = _parse_choice_response(content, list(q["criteria"].keys()))
                    if probs:
                        results[idx] = LLMLabelResult(
                            item_id=item["id"],
                            question_type=q_type,
                            teacher_probs=probs,
                            teacher_label=choice,
                            model=model,
                        )
                        return

                elif q_type == "score":
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
