"""Augment binary noul questions with multi-option alternatives.

Expands true/false noul questions to 5 options using cognitive-type-specific
templates (deterministic, free) or LLM-generated confounders (fallback).
This balances option counts across question types for unified-head training.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

LLM_AUGMENT_MODEL_ENV = "LLM_AUGMENT_MODEL"
DEFAULT_AUGMENT_MODEL = "argo:gpt-4.1-nano"

COGNITIVE_TYPE_AUGMENT_TEMPLATES: dict[str, dict[str, str]] = {
    "entailment": {
        "true": "The evidence fully supports the claim",
        "false": "The evidence contradicts the claim",
        "partially_supported": "The evidence partially supports but does not fully confirm",
        "neutral": "The evidence is unrelated to the claim",
        "insufficient_evidence": "Not enough evidence to make a determination",
    },
    "fact_verification": {
        "true": "The stated fact is accurate given the evidence",
        "false": "The stated fact is inaccurate given the evidence",
        "partially_true": "The fact is partly correct but contains inaccuracies",
        "unverifiable": "The fact cannot be verified from available information",
        "outdated": "The fact may have been true but is no longer current",
    },
    "comparison": {
        "true": "The comparison holds as stated",
        "false": "The comparison does not hold as stated",
        "approximately_equal": "The compared quantities are roughly equivalent",
        "depends_on_metric": "The result varies depending on measurement criteria",
        "insufficient_data": "Not enough data to make a reliable comparison",
    },
    "temporal": {
        "true": "The temporal relationship holds as described",
        "false": "The temporal relationship does not hold as described",
        "simultaneous": "The events occurred at approximately the same time",
        "ambiguous_timeline": "The timeline is unclear or can be interpreted differently",
        "insufficient_temporal_data": "Not enough temporal information to determine ordering",
    },
    "causal": {
        "true": "A direct causal relationship exists as stated",
        "false": "No causal relationship exists as stated",
        "contributing_factor": "A partial or indirect causal influence exists",
        "correlation_not_causation": "A correlation exists but causation is not established",
        "insufficient_evidence": "Not enough evidence to determine causality",
    },
    "sufficiency": {
        "true": "The evidence is sufficient to support the conclusion",
        "false": "The evidence is not sufficient to support the conclusion",
        "borderline": "The evidence is marginally sufficient but not robust",
        "sufficient_with_caveats": "Sufficient under certain assumptions or conditions",
        "more_info_needed": "Additional specific evidence would be required",
    },
    "consistency": {
        "true": "The statements are logically consistent with each other",
        "false": "The statements contain a logical contradiction",
        "mostly_consistent": "Minor inconsistencies exist but do not affect the main point",
        "trivially_inconsistent": "A technical inconsistency exists but is inconsequential",
        "cannot_determine": "Consistency cannot be assessed from available information",
    },
    "possibility": {
        "true": "The scenario is feasible under the given constraints",
        "false": "The scenario is not feasible under the given constraints",
        "theoretically_possible": "Possible in theory but unlikely in practice",
        "conditionally_possible": "Possible only if specific additional conditions are met",
        "practically_infeasible": "Technically possible but impractical to achieve",
    },
    "classification": {
        "true": "The item clearly belongs to the specified category",
        "false": "The item clearly does not belong to the specified category",
        "borderline": "The item falls on the boundary between categories",
        "overlapping_categories": "The item belongs to multiple categories simultaneously",
        "insufficient_criteria": "The category criteria are too vague to classify definitively",
    },
    "safety": {
        "true": "A significant safety risk or concern exists",
        "false": "No meaningful safety risk or concern exists",
        "minor_risk": "A low-level risk exists but is manageable",
        "context_dependent": "Safety depends on specific circumstances or usage patterns",
        "insufficient_safety_data": "Not enough information to assess safety reliably",
    },
    "counterfactual": {
        "true": "The counterfactual outcome would occur as described",
        "false": "The counterfactual outcome would not occur as described",
        "partially_different": "Some but not all aspects of the outcome would change",
        "unpredictable": "The counterfactual outcome cannot be reliably predicted",
        "depends_on_assumptions": "The outcome varies depending on which assumptions hold",
    },
    "threshold": {
        "true": "The value clearly exceeds the defined threshold",
        "false": "The value clearly falls below the defined threshold",
        "borderline": "The value is at or very near the threshold boundary",
        "approaching": "The value is trending toward the threshold but has not reached it",
        "measurement_uncertain": "Measurement uncertainty makes threshold comparison unreliable",
    },
}

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def match_template(noul_question: dict[str, Any]) -> dict[str, str] | None:
    """Return augmented options template for a noul question, or None."""
    ct = noul_question.get("cognitive_type", "")
    if ct and ct in COGNITIVE_TYPE_AUGMENT_TEMPLATES:
        return COGNITIVE_TYPE_AUGMENT_TEMPLATES[ct]
    return None


def _build_augment_prompt(
    state: str, instructions: str, label: bool, cognitive_type: str
) -> str:
    return (
        f"You are expanding a binary (true/false) decision question into a "
        f"multi-option question by generating plausible alternative answers.\n\n"
        f"Context:\n"
        f"  State: {state[:800]}\n"
        f"  Question: {instructions}\n"
        f"  Correct answer: {'true' if label else 'false'}\n"
        f"  Cognitive type: {cognitive_type}\n\n"
        f"Generate exactly 3 additional answer options that are:\n"
        f"- Plausible but distinguishably different from true/false\n"
        f"- Semantically relevant to the question domain\n"
        f"- Meaningful gradations (e.g. partial, conditional, uncertain)\n\n"
        f"Rules:\n"
        f"- Keys must be snake_case identifiers\n"
        f"- Do NOT include 'true' or 'false' as keys\n"
        f"- Each description should be 5-15 words\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"additional_options": {{"<key1>": "<description>", "<key2>": "<description>", "<key3>": "<description>"}}}}'
    )


def parse_augment_response(content: str) -> dict[str, str] | None:
    """Parse LLM response into additional options dict, or None on failure."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    additional = parsed.get("additional_options")
    if not isinstance(additional, dict):
        return None

    # Filter to valid snake_case keys, exclude true/false
    filtered = {
        k: str(v)
        for k, v in additional.items()
        if isinstance(k, str)
        and _SNAKE_RE.match(k)
        and k not in ("true", "false")
        and isinstance(v, str)
        and v.strip()
    }

    if len(filtered) < 2:
        return None

    return _build_full_options(filtered)


def _build_full_options(additional: dict[str, str]) -> dict[str, str]:
    """Combine LLM-generated options with standard true/false entries."""
    base = {
        "true": "The proposition is correct and supported by available evidence",
        "false": "The proposition is incorrect or contradicted by available evidence",
    }
    base.update(additional)
    return base


async def augment_noul_questions(
    client: Any | None,
    families: list[dict[str, Any]],
    model: str,
    base_url: str,
    max_concurrent: int = 10,
) -> tuple[list[dict[str, Any]], int, int]:
    """Augment noul questions in families with multi-option alternatives.

    Args:
        client: Async HTTP client (None for template-only mode).
        families: Family dicts to augment in place.
        model: LLM model identifier for fallback generation.
        base_url: LLM API base URL.
        max_concurrent: Max concurrent LLM requests.

    Returns:
        (families, augmented_count, failed_count).
    """
    augmented = 0
    failed = 0
    llm_queue: list[tuple[dict, str, str, bool, str]] = []

    # Phase 1: apply templates, queue LLM fallbacks
    for family in families:
        state = family.get("state", "")
        for nq in family.get("noul_questions", []):
            if nq.get("augmented_options"):
                continue

            template = match_template(nq)
            if template is not None:
                nq["augmented_options"] = template
                nq["_augment_source"] = "template"
                augmented += 1
            else:
                ct = nq.get("cognitive_type", "unknown")
                instructions = nq.get("instructions", "")
                label = nq.get("label", True)
                llm_queue.append((nq, state, instructions, label, ct))

    if not llm_queue:
        return families, augmented, failed

    # Phase 2: LLM fallback for unmatched questions
    if client is None:
        logger.warning(
            "No LLM client available; %d noul questions without templates skipped",
            len(llm_queue),
        )
        return families, augmented, len(llm_queue)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _augment_one(
        nq: dict, state: str, instructions: str, label: bool, ct: str
    ) -> bool:
        prompt = _build_augment_prompt(state, instructions, label, ct)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
        }

        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.post(
                        f"{base_url}/v1/chat/completions", json=payload
                    )
                if resp.status_code == 429:
                    await asyncio.sleep(2**attempt)
                    continue
                if resp.status_code != 200:
                    logger.warning(
                        "Augment API error %d (attempt %d)", resp.status_code, attempt + 1
                    )
                    await asyncio.sleep(1)
                    continue

                content = resp.json()["choices"][0]["message"]["content"]
                options = parse_augment_response(content)
                if options:
                    nq["augmented_options"] = options
                    nq["_augment_source"] = "llm"
                    return True

            except Exception as exc:
                logger.warning("Augment failed (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)

        return False

    tasks = [
        _augment_one(nq, state, instructions, label, ct)
        for nq, state, instructions, label, ct in llm_queue
    ]

    for coro in asyncio.as_completed(tasks):
        success = await coro
        if success:
            augmented += 1
        else:
            failed += 1

    return families, augmented, failed
