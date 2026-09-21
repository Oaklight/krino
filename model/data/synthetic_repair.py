"""Repair invalid choice/score labels in variant data via cheap LLM calls."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

LLM_REPAIR_MODEL_ENV = "LLM_REPAIR_MODEL"
DEFAULT_REPAIR_MODEL = "argo:gpt-4.1-nano"


def _find_bad_labels(
    families: list[dict],
    variants: list[dict],
    domain_template: dict,
) -> list[dict]:
    """Scan for choice/score labels that don't match criteria keys."""
    issues: list[dict] = []

    for fi, (family, variant) in enumerate(zip(families, variants)):
        choice_qs = family.get("choice_questions", [])
        if not choice_qs and family.get("choice_question"):
            choice_qs = [family["choice_question"]]
        score_qs = family.get("score_questions", [])
        if not score_qs and family.get("score_question"):
            score_qs = [family["score_question"]]

        # Check counterfactual choice label
        cf = variant.get("counterfactual")
        if cf and cf.get("choice_label") and choice_qs:
            first_cq = choice_qs[0]
            criteria = first_cq.get("criteria", domain_template["choice_template"]["criteria"])
            label = cf["choice_label"]
            if label not in criteria:
                issues.append({
                    "family_idx": fi,
                    "variant_type": "counterfactual",
                    "field": "choice_label",
                    "bad_label": label,
                    "valid_keys": list(criteria.keys()),
                    "state": cf.get("state", family.get("state", "")),
                    "instructions": first_cq.get("instructions", domain_template["choice_template"]["instructions"]),
                    "criteria": criteria,
                })

        # Check counterfactual score label
        if cf and cf.get("score_label") is not None and score_qs:
            first_sq = score_qs[0]
            criteria = first_sq.get("criteria", domain_template["score_template"]["criteria"])
            score_label = float(cf["score_label"])
            if not (1.0 <= score_label <= float(len(criteria))):
                issues.append({
                    "family_idx": fi,
                    "variant_type": "counterfactual",
                    "field": "score_label",
                    "bad_label": score_label,
                    "valid_range": (1.0, float(len(criteria))),
                    "state": cf.get("state", family.get("state", "")),
                    "instructions": first_sq.get("instructions", domain_template["score_template"]["instructions"]),
                    "criteria": criteria,
                })

        # Check base choice labels
        for ci, cq in enumerate(choice_qs):
            criteria = cq.get("criteria", domain_template["choice_template"]["criteria"])
            if cq["label"] not in criteria:
                issues.append({
                    "family_idx": fi,
                    "variant_type": "base",
                    "field": f"choice_questions[{ci}].label",
                    "bad_label": cq["label"],
                    "valid_keys": list(criteria.keys()),
                    "state": family.get("state", ""),
                    "instructions": cq.get("instructions", ""),
                    "criteria": criteria,
                })

    return issues


async def _repair_choice_label(
    client: Any,
    issue: dict,
    model: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Ask a cheap LLM to pick the correct key."""
    keys = issue["valid_keys"]
    key_list = ", ".join(f'"{k}"' for k in keys)

    prompt = (
        f"State: {issue['state'][:500]}\n\n"
        f"Question: {issue['instructions']}\n\n"
        f"Valid option keys: [{key_list}]\n\n"
        f"Pick the single best option key. Reply with ONLY the key, nothing else."
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 20,
    }

    async with semaphore:
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                logger.warning("Repair API error %d", resp.status_code)
                return None
            content = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
            if content in keys:
                return content
            for k in keys:
                if k.lower() == content.lower():
                    return k
            logger.warning("Repair returned %r, not in %s", content, keys)
            return None
        except Exception as exc:
            logger.warning("Repair failed: %s", exc)
            return None


async def _repair_score_label(
    client: Any,
    issue: dict,
    model: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> float | None:
    """Ask a cheap LLM to pick a valid score."""
    lo, hi = issue["valid_range"]
    criteria = issue["criteria"]
    criteria_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    prompt = (
        f"State: {issue['state'][:500]}\n\n"
        f"Question: {issue['instructions']}\n\n"
        f"Score levels:\n{criteria_list}\n\n"
        f"Pick a score from {lo:.0f} to {hi:.0f}. Reply with ONLY the number."
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }

    async with semaphore:
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                return None
            content = resp.json()["choices"][0]["message"]["content"].strip()
            val = float(content)
            if lo <= val <= hi:
                return val
            return None
        except Exception:
            return None


async def repair_variants(
    client: Any,
    families: list[dict],
    variants: list[dict],
    domain_template: dict,
    model: str,
    base_url: str,
    max_concurrent: int = 10,
) -> tuple[list[dict], int, int]:
    """Repair bad labels in variants. Returns (updated_variants, repaired, failed)."""
    issues = _find_bad_labels(families, variants, domain_template)
    if not issues:
        logger.info("  No bad labels to repair")
        return variants, 0, 0

    logger.info("  Found %d bad labels to repair", len(issues))
    semaphore = asyncio.Semaphore(max_concurrent)
    repaired = 0
    failed = 0

    for issue in issues:
        fi = issue["family_idx"]

        if issue["field"] == "choice_label":
            fixed = await _repair_choice_label(client, issue, model, base_url, semaphore)
            if fixed:
                variants[fi]["counterfactual"]["choice_label"] = fixed
                repaired += 1
                logger.info(
                    "  Repaired family %d cf-choice: %r → %r",
                    fi, issue["bad_label"], fixed,
                )
            else:
                failed += 1

        elif issue["field"] == "score_label":
            fixed = await _repair_score_label(client, issue, model, base_url, semaphore)
            if fixed:
                variants[fi]["counterfactual"]["score_label"] = fixed
                repaired += 1
                logger.info(
                    "  Repaired family %d cf-score: %.1f → %.1f",
                    fi, issue["bad_label"], fixed,
                )
            else:
                failed += 1

        elif issue["field"].startswith("choice_questions["):
            fixed = await _repair_choice_label(client, issue, model, base_url, semaphore)
            if fixed:
                idx = int(issue["field"].split("[")[1].split("]")[0])
                cq_key = "choice_questions" if "choice_questions" in families[fi] else "choice_question"
                if cq_key == "choice_questions":
                    families[fi]["choice_questions"][idx]["label"] = fixed
                else:
                    families[fi]["choice_question"]["label"] = fixed
                repaired += 1
                logger.info(
                    "  Repaired family %d base %s: %r → %r",
                    fi, issue["field"], issue["bad_label"], fixed,
                )
            else:
                failed += 1

    logger.info("  Repair complete: %d fixed, %d unfixable", repaired, failed)
    return variants, repaired, failed
