"""Cross-model validation for synthetic data items."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


@dataclass
class ValidationResult:
    correct: bool
    confidence: float
    reasoning: str


async def validate_item(
    client: Any,
    state: str,
    instructions: str,
    label: str,
    model: str,
    base_url: str,
) -> ValidationResult:
    """Validate a single item's label using a judge model."""
    from .synthetic_templates import build_validation_prompt

    prompt = build_validation_prompt(state, instructions, str(label))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Validation rate limited, waiting %ds...", wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning("Validation API error %d (attempt %d): %s", resp.status_code, attempt + 1, resp.text[:200])
                await asyncio.sleep(1)
                continue

            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            return ValidationResult(
                correct=parsed.get("correct", False),
                confidence=parsed.get("confidence", 0.0),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse validation response (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("Validation request failed (attempt %d): %s", attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)

    return ValidationResult(correct=False, confidence=0.0, reasoning="All retries exhausted")


async def validate_family(
    client: Any,
    family: dict[str, Any],
    model: str,
    base_url: str,
    min_agreement: int = 2,
    num_votes: int = 3,
) -> tuple[bool, list[dict]]:
    """Validate all items in a family with majority-vote checking.

    Returns (passed, details) where details is a list of per-item results.
    """
    state = family["state"]
    items_to_check: list[tuple[str, str]] = []

    for nq in family.get("noul_questions", []):
        items_to_check.append((nq["instructions"], str(nq["label"])))

    for cq in family.get("choice_questions", []):
        items_to_check.append((cq["instructions"], cq["label"]))

    for sq in family.get("score_questions", []):
        items_to_check.append((sq["instructions"], str(sq["label"])))

    details = []
    all_passed = True

    for instructions, label in items_to_check:
        votes = await asyncio.gather(*[
            validate_item(client, state, instructions, label, model, base_url)
            for _ in range(num_votes)
        ])
        agrees = sum(1 for v in votes if v.correct)
        passed = agrees >= min_agreement
        if not passed:
            all_passed = False
        details.append({
            "instructions": instructions,
            "label": label,
            "votes": agrees,
            "total": num_votes,
            "passed": passed,
            "reasoning": [v.reasoning for v in votes],
        })

    return all_passed, details
