"""Synthetic data generation pipeline for typed decision training.

Hybrid approach:
- Seeded domains: real states from existing benchmarks + LLM-generated questions
- Pure generation domains: LLM generates both states and questions
- All items get Jev API soft labels

Composable stages — run all at once or independently:
    python -m model.data.synthetic --stages base                     # families only
    python -m model.data.synthetic --stages counterfactual,paraphrase,negation  # variants
    python -m model.data.synthetic --stages validate                 # cross-model check
    python -m model.data.synthetic --stages jev-label                # Jev soft labels
    python -m model.data.synthetic                                   # all stages

Each stage reads/writes to DATA_DIR. Stages can be rerun independently.
Resume: cached families/variants on disk are reused automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "probing" / "scripts"))
sys.path.insert(0, str(_repo_root / "_vendor"))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env")

from .format import TypedQuestion
from .synthetic_label import LabelResult, label_items_sync, print_bucket_report
from .synthetic_templates import (
    COGNITIVE_TYPE_DESCRIPTIONS,
    DOMAIN_TEMPLATES,
    FAMILIES_PER_DOMAIN,
    GENERATED_DOMAINS,
    NOUL_QUESTIONS_PER_FAMILY,
    SEEDED_DOMAINS,
    build_generated_family_prompt,
    build_seeded_family_prompt,
    build_variant_prompt,
    json_compact,
)
from .lsh import LSHIndex
from .synthetic_dedup import dedup_families
from .synthetic_repair import repair_variants
from .synthetic_validate import validate_family

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "benchmarks" / "synthetic"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_GEN_MODEL = os.environ.get("LLM_GEN_MODEL", "claude-sonnet-4-20250514")
LLM_VARIANT_MODEL = os.environ.get("LLM_VARIANT_MODEL", "GPT-4.1-mini")
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gemini-2.0-flash")
LLM_REPAIR_MODEL = os.environ.get("LLM_REPAIR_MODEL", "argo:gpt-4.1-nano")

MAX_CONCURRENT = int(os.environ.get("SYNTH_MAX_CONCURRENT", "20"))
MAX_RETRIES = 3

ALL_STAGES = ["base", "counterfactual", "paraphrase", "negation", "shuffle", "repair", "dedup", "validate", "jev-label", "report"]
VARIANT_STAGES = {"counterfactual", "paraphrase", "negation"}


# --- File I/O helpers ---

# probing/scripts contains the zerodep httpclient (async HTTP) and jev_client
# (Jev API). They are path-injected here because they are standalone scripts,
# not installable packages.


def _save_jsonl(items: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def _append_jsonl(item: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- LLM helpers ---


async def _llm_chat(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str | None:
    """Call LLM via OpenAI-compatible endpoint and return the content string."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                f"{LLM_BASE_URL}/v1/chat/completions", json=payload
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited, waiting %ds...", wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning(
                    "LLM API error %d (attempt %d): %s",
                    resp.status_code,
                    attempt + 1,
                    resp.text[:200],
                )
                await asyncio.sleep(1)
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as exc:
            logger.warning("LLM request failed (attempt %d): %s", attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    return None


def _parse_json_response(content: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code fences."""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s — %.200s", exc, text)
        return None


# --- Seed loading ---


def _load_seed_states(domain: str, count: int, seed: int = 42) -> list[str]:
    """Load real states from an existing benchmark for seeding."""
    template = DOMAIN_TEMPLATES[domain]
    source = template.get("seed_source")
    if not source:
        return []

    from .pipeline import LOADERS

    loader = LOADERS.get(source)
    if not loader:
        logger.warning("Seed source %s not found in LOADERS", source)
        return []

    rng = random.Random(seed)
    items = list(loader())
    train_items = [item for item in items if item.split == "train"]
    if not train_items:
        train_items = items

    states = sorted({item.state for item in train_items})
    rng.shuffle(states)
    states = [s for s in states if len(s) >= 50]

    return states[:count]


# --- Stage: base family generation ---


async def generate_seeded_family(
    client: Any,
    domain: str,
    seed_state: str,
    cognitive_types: list[str],
    family_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        prompt = build_seeded_family_prompt(domain, seed_state, cognitive_types)
        content = await _llm_chat(client, LLM_GEN_MODEL, prompt)
        parsed = _parse_json_response(content)
        if not parsed:
            return None
        parsed["state"] = seed_state
        parsed["_domain"] = domain
        parsed["_family_idx"] = family_idx
        parsed["_seeded"] = True
        return parsed


async def generate_pure_family(
    client: Any,
    domain: str,
    cognitive_types: list[str],
    family_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        prompt = build_generated_family_prompt(domain, cognitive_types)
        content = await _llm_chat(client, LLM_GEN_MODEL, prompt)
        parsed = _parse_json_response(content)
        if not parsed:
            return None
        parsed["_domain"] = domain
        parsed["_family_idx"] = family_idx
        parsed["_seeded"] = False
        return parsed


async def run_base_stage(
    client: Any,
    domains: list[str],
    families_per_domain: int,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, list[dict]]:
    """Generate base families per domain. Resumes from cached families on disk."""
    rng = random.Random(seed)
    all_cog_types = list(COGNITIVE_TYPE_DESCRIPTIONS.keys())
    result: dict[str, list[dict]] = {}

    for domain in domains:
        template = DOMAIN_TEMPLATES[domain]
        is_seeded = template.get("seed_source") is not None
        t0 = time.monotonic()

        logger.info("=" * 60)
        logger.info("Domain: %s (%s)", domain, "seeded" if is_seeded else "generated")
        logger.info("=" * 60)

        # Check for cached families
        families_path = DATA_DIR / f"{domain}_families.jsonl"
        cached = _load_jsonl(families_path)
        if len(cached) >= families_per_domain:
            logger.info("  Resuming: %d families already cached, skipping base generation", len(cached))
            result[domain] = cached[:families_per_domain]
            # Advance rng to keep determinism
            for _ in range(families_per_domain):
                rng.sample(all_cog_types, min(NOUL_QUESTIONS_PER_FAMILY, len(all_cog_types)))
            continue

        seed_states: list[str] = []
        if is_seeded:
            logger.info("  Loading seed states from %s...", template["seed_source"])
            try:
                seed_states = _load_seed_states(domain, families_per_domain, seed)
                logger.info("  Loaded %d seed states", len(seed_states))
            except Exception as exc:
                logger.warning("  Failed to load seeds: %s, falling back to pure generation", exc)

        actual_count = min(
            families_per_domain,
            len(seed_states) if seed_states else families_per_domain,
        )

        # Resume: generate only missing families
        start_idx = len(cached)
        if start_idx > 0:
            logger.info("  Resuming from family %d (have %d cached)", start_idx, start_idx)
            # Advance rng past cached families
            for _ in range(start_idx):
                rng.sample(all_cog_types, min(NOUL_QUESTIONS_PER_FAMILY, len(all_cog_types)))

        remaining = actual_count - start_idx
        if remaining <= 0:
            result[domain] = cached[:actual_count]
            continue

        logger.info("  Generating %d families (%d-%d)...", remaining, start_idx, actual_count - 1)

        # Build LSH index from cached families for on-the-fly dedup
        dedup_threshold = 0.7
        lsh = LSHIndex(num_perm=128, bands=16)
        dedup_sigs = []
        for ci, cf in enumerate(cached):
            mh = lsh.make_minhash(cf.get("state", ""))
            lsh.insert(ci, mh.signature)
            dedup_sigs.append(mh)
        dedup_dropped = 0
        _dedup_lock = asyncio.Lock()

        async def _generate_and_save(idx: int, cog_types: list[str]) -> dict | None:
            nonlocal dedup_dropped
            if seed_states:
                family = await generate_seeded_family(
                    client, domain, seed_states[idx], cog_types, idx, semaphore
                )
            else:
                family = await generate_pure_family(
                    client, domain, cog_types, idx, semaphore
                )
            if not family:
                return None

            # On-the-fly dedup check
            async with _dedup_lock:
                mh = lsh.make_minhash(family.get("state", ""))
                candidates = lsh.query(mh.signature)
                is_dup = any(
                    mh.jaccard(dedup_sigs[c]) >= dedup_threshold
                    for c in candidates
                )
                if is_dup:
                    dedup_dropped += 1
                    logger.info("  Dedup: family %d is near-duplicate, skipping", idx)
                    return None
                sig_idx = len(dedup_sigs)
                lsh.insert(sig_idx, mh.signature)
                dedup_sigs.append(mh)

            _append_jsonl(family, families_path)
            return family

        tasks = []
        for i in range(start_idx, actual_count):
            cog_types = rng.sample(
                all_cog_types, min(NOUL_QUESTIONS_PER_FAMILY, len(all_cog_types))
            )
            tasks.append(_generate_and_save(i, cog_types))

        new_families = await asyncio.gather(*tasks)
        new_valid = [f for f in new_families if f is not None]
        all_families = cached + new_valid
        result[domain] = all_families

        if dedup_dropped:
            logger.info("  Dedup: dropped %d near-duplicate families during generation", dedup_dropped)

        total_valid = len(all_families)
        logger.info(
            "  Done: %d/%d families (%.1fs, %d new, %d cached)",
            total_valid, actual_count, time.monotonic() - t0, len(new_valid), len(cached),
        )

    return result


# --- Stage: variant generation (counterfactual / paraphrase / negation) ---


async def _gen_single_variant(
    client: Any,
    family: dict,
    variant_type: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    family_json = json.dumps(
        {k: v for k, v in family.items() if not k.startswith("_")},
        ensure_ascii=False,
        indent=2,
    )
    async with semaphore:
        prompt = build_variant_prompt(family_json, variant_type)
        content = await _llm_chat(client, LLM_VARIANT_MODEL, prompt)
        return _parse_json_response(content)


async def run_variant_stage(
    client: Any,
    families_by_domain: dict[str, list[dict]],
    variant_types: set[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[dict[str, dict | None]]]:
    """Generate requested variant types for each family. Resumes from cache."""
    result: dict[str, list[dict]] = {}

    for domain, families in families_by_domain.items():
        t0 = time.monotonic()
        ordered_types = sorted(variant_types & VARIANT_STAGES)

        # Load cached variants
        variants_path = DATA_DIR / f"{domain}_variants.jsonl"
        cached_variants = _load_jsonl(variants_path)
        start_idx = len(cached_variants)

        if start_idx >= len(families):
            logger.info("  %s: all %d variant sets cached, skipping", domain, len(families))
            result[domain] = cached_variants[:len(families)]
            continue

        if start_idx > 0:
            logger.info(
                "  %s: resuming variants from family %d (have %d cached)",
                domain, start_idx, start_idx,
            )

        remaining = len(families) - start_idx
        logger.info(
            "  Generating variants (%s) for %d families in %s...",
            ", ".join(ordered_types), remaining, domain,
        )

        async def _gen_family_variants(fi: int) -> tuple[int, dict]:
            family = families[fi]
            tasks = {
                vt: _gen_single_variant(client, family, vt, semaphore)
                for vt in ordered_types
            }
            results = await asyncio.gather(*tasks.values())
            return fi, dict(zip(tasks.keys(), results))

        # Fan out across all remaining families
        gen_tasks = [_gen_family_variants(fi) for fi in range(start_idx, len(families))]
        pending = {asyncio.ensure_future(t): t for t in gen_tasks}
        results_by_idx: dict[int, dict] = {}
        next_to_write = start_idx
        domain_variants = list(cached_variants)
        done_count = 0

        for coro in asyncio.as_completed(pending):
            fi, variant_data = await coro
            results_by_idx[fi] = variant_data
            done_count += 1

            # Write in order to preserve resume correctness
            while next_to_write in results_by_idx:
                domain_variants.append(results_by_idx.pop(next_to_write))
                _append_jsonl(domain_variants[-1], variants_path)
                next_to_write += 1

            if done_count % 10 == 0 or done_count == remaining:
                logger.info(
                    "  %s: variants %d/%d done, %d written (%.1fs)",
                    domain, done_count, remaining, next_to_write, time.monotonic() - t0,
                )

        result[domain] = domain_variants
        logger.info("  %s: variants complete (%.1fs)", domain, time.monotonic() - t0)

    return result


# --- Stage: validate ---


async def run_validate_stage(
    client: Any,
    families_by_domain: dict[str, list[dict]],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[dict]]:
    """Cross-model validation. Returns filtered families_by_domain."""
    result: dict[str, list[dict]] = {}

    for domain, families in families_by_domain.items():
        logger.info("  Validating %d families in %s...", len(families), domain)
        val_tasks = [
            validate_family(client, f, LLM_JUDGE_MODEL, LLM_BASE_URL)
            for f in families
        ]
        val_results = await asyncio.gather(*val_tasks)
        valid = [f for f, (passed, _) in zip(families, val_results) if passed]
        rejected = len(families) - len(valid)
        result[domain] = valid
        logger.info("  %d/%d passed (%d rejected)", len(valid), len(families), rejected)

    return result


# --- Convert to TypedQuestion ---


def family_to_typed_questions(
    family: dict,
    variants: dict[str, dict | None],
    domain: str,
    family_idx: int,
    enabled_stages: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a family + variants into serializable item dicts.

    If enabled_stages is given, only emit item types for those stages.
    """
    state = family["state"]
    group = f"synthetic-{domain}-{family_idx:04d}"
    items: list[dict[str, Any]] = []

    template = DOMAIN_TEMPLATES[domain]
    do_all = enabled_stages is None
    do_base = do_all or "base" in enabled_stages

    # Base noul questions
    if do_base:
        for nq in family.get("noul_questions", []):
            ct = nq.get("cognitive_type", "unknown")
            items.append(
                TypedQuestion.noul(
                    id=f"synthetic-{domain}-{family_idx:04d}-noul-{ct}",
                    state=state,
                    instructions=nq["instructions"],
                    label=nq["label"],
                    source="synthetic",
                    split="train",
                    group=group,
                ).to_dict()
            )

    # Base choice questions (array or single for backward compat)
    choice_questions_raw = family.get("choice_questions", [])
    if not choice_questions_raw and family.get("choice_question"):
        choice_questions_raw = [family["choice_question"]]
    choice_questions_valid = []
    for ci, cq in enumerate(choice_questions_raw):
        cq_criteria = cq.get("criteria", template["choice_template"]["criteria"])
        cq_label = cq["label"]
        if cq_label in cq_criteria:
            if do_base:
                items.append(
                    TypedQuestion.choice(
                        id=f"synthetic-{domain}-{family_idx:04d}-choice-{ci}",
                        state=state,
                        instructions=cq.get("instructions", template["choice_template"]["instructions"]),
                        criteria=cq_criteria,
                        label=cq_label,
                        source="synthetic",
                        split="train",
                        group=group,
                    ).to_dict()
                )
            choice_questions_valid.append(cq)
        else:
            logger.warning(
                "Dropped choice-%d for %s-%04d: label %r not in criteria keys %s",
                ci, domain, family_idx, cq_label, sorted(cq_criteria.keys()),
            )

    # Base score questions (array or single for backward compat)
    score_questions_raw = family.get("score_questions", [])
    if not score_questions_raw and family.get("score_question"):
        score_questions_raw = [family["score_question"]]
    score_questions_valid = []
    for si, sq in enumerate(score_questions_raw):
        sq_criteria = sq.get("criteria", template["score_template"]["criteria"])
        sq_label_raw = float(sq["label"])
        if 1.0 <= sq_label_raw <= float(len(sq_criteria)):
            sq_label = sq_label_raw - 1.0  # convert 1-based LLM output to 0-based training index
            if do_base:
                items.append(
                    TypedQuestion.score(
                        id=f"synthetic-{domain}-{family_idx:04d}-score-{si}",
                        state=state,
                        instructions=sq.get("instructions", template["score_template"]["instructions"]),
                        criteria=sq_criteria,
                        label=sq_label,
                        source="synthetic",
                        split="train",
                        group=group,
                    ).to_dict()
                )
            score_questions_valid.append({**sq, "_label_0based": sq_label})
        else:
            logger.warning(
                "Dropped score-%d for %s-%04d: label %.1f out of range [1, %d]",
                si, domain, family_idx, sq_label_raw, len(sq_criteria),
            )

    # --- Variants ---

    # Counterfactual
    if do_all or "counterfactual" in enabled_stages:
        cf = variants.get("counterfactual")
        if cf and cf.get("state"):
            cf_state = cf["state"]
            cf_noul_labels = {
                nl["cognitive_type"]: nl["label"]
                for nl in cf.get("noul_labels", [])
            }
            for nq in family.get("noul_questions", []):
                ct = nq.get("cognitive_type", "unknown")
                new_label = cf_noul_labels.get(ct, nq["label"])
                items.append(
                    TypedQuestion.noul(
                        id=f"synthetic-{domain}-{family_idx:04d}-cf-noul-{ct}",
                        state=cf_state,
                        instructions=nq["instructions"],
                        label=new_label,
                        source="synthetic",
                        split="train",
                        group=group,
                    ).to_dict()
                )
            if choice_questions_valid and cf.get("choice_label"):
                first_cq = choice_questions_valid[0]
                cf_choice_criteria = first_cq.get("criteria", template["choice_template"]["criteria"])
                cf_choice_label = cf["choice_label"]
                if cf_choice_label in cf_choice_criteria:
                    items.append(
                        TypedQuestion.choice(
                            id=f"synthetic-{domain}-{family_idx:04d}-cf-choice-0",
                            state=cf_state,
                            instructions=first_cq.get("instructions", template["choice_template"]["instructions"]),
                            criteria=cf_choice_criteria,
                            label=cf_choice_label,
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )
                else:
                    logger.warning(
                        "Dropped cf-choice for %s-%04d: label %r not in criteria keys %s",
                        domain, family_idx, cf_choice_label, sorted(cf_choice_criteria.keys()),
                    )
            if score_questions_valid and cf.get("score_label") is not None:
                first_sq = score_questions_valid[0]
                cf_score_label_raw = float(cf["score_label"])
                cf_score_criteria = first_sq.get("criteria", template["score_template"]["criteria"])
                if 1.0 <= cf_score_label_raw <= float(len(cf_score_criteria)):
                    items.append(
                        TypedQuestion.score(
                            id=f"synthetic-{domain}-{family_idx:04d}-cf-score-0",
                            state=cf_state,
                            instructions=first_sq.get("instructions", template["score_template"]["instructions"]),
                            criteria=cf_score_criteria,
                            label=cf_score_label_raw - 1.0,
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )
                else:
                    logger.warning(
                        "Dropped cf-score for %s-%04d: label %.1f out of range [1, %d]",
                        domain, family_idx, cf_score_label_raw, len(cf_score_criteria),
                    )

    # Paraphrase
    if do_all or "paraphrase" in enabled_stages:
        para = variants.get("paraphrase")
        if para:
            para_state = para.get("state_paraphrase")
            if para_state:
                for nq in family.get("noul_questions", []):
                    ct = nq.get("cognitive_type", "unknown")
                    items.append(
                        TypedQuestion.noul(
                            id=f"synthetic-{domain}-{family_idx:04d}-para-state-noul-{ct}",
                            state=para_state,
                            instructions=nq["instructions"],
                            label=nq["label"],
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )
                for ci, cq in enumerate(choice_questions_valid):
                    items.append(
                        TypedQuestion.choice(
                            id=f"synthetic-{domain}-{family_idx:04d}-para-state-choice-{ci}",
                            state=para_state,
                            instructions=cq["instructions"],
                            criteria=cq["criteria"],
                            label=cq["label"],
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )
                for si, sq in enumerate(score_questions_valid):
                    items.append(
                        TypedQuestion.score(
                            id=f"synthetic-{domain}-{family_idx:04d}-para-state-score-{si}",
                            state=para_state,
                            instructions=sq["instructions"],
                            criteria=sq["criteria"],
                            label=sq["_label_0based"],
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )

            q_paras = {
                qp["original_instructions"]: qp["paraphrased_instructions"]
                for qp in para.get("question_paraphrases", [])
            }
            if q_paras:
                for nq in family.get("noul_questions", []):
                    ct = nq.get("cognitive_type", "unknown")
                    new_instr = q_paras.get(nq["instructions"])
                    if new_instr:
                        items.append(
                            TypedQuestion.noul(
                                id=f"synthetic-{domain}-{family_idx:04d}-para-q-noul-{ct}",
                                state=state,
                                instructions=new_instr,
                                label=nq["label"],
                                source="synthetic",
                                split="train",
                                group=group,
                            ).to_dict()
                        )

    # Option-order shuffle
    if do_all or "shuffle" in enabled_stages:
        rng = random.Random(f"{domain}-{family_idx}")
        for ci, cq in enumerate(choice_questions_valid):
            keys = list(cq["criteria"].keys())
            if len(keys) >= 3:
                shuffled_keys = keys[:]
                rng.shuffle(shuffled_keys)
                if shuffled_keys != keys:
                    shuffled_criteria = {k: cq["criteria"][k] for k in shuffled_keys}
                    items.append(
                        TypedQuestion.choice(
                            id=f"synthetic-{domain}-{family_idx:04d}-shuffle-choice-{ci}",
                            state=state,
                            instructions=cq["instructions"],
                            criteria=shuffled_criteria,
                            label=cq["label"],
                            source="synthetic",
                            split="train",
                            group=group,
                        ).to_dict()
                    )

    # Negation
    if do_all or "negation" in enabled_stages:
        neg = variants.get("negation")
        if neg:
            for nq_neg in neg.get("negated_questions", []):
                ct = nq_neg.get("cognitive_type", "unknown")
                items.append(
                    TypedQuestion.noul(
                        id=f"synthetic-{domain}-{family_idx:04d}-neg-noul-{ct}",
                        state=state,
                        instructions=nq_neg["negated_instructions"],
                        label=nq_neg["negated_label"],
                        source="synthetic",
                        split="train",
                        group=group,
                    ).to_dict()
                )

    return items


# --- Main pipeline ---


async def run_pipeline(
    domains: list[str] | None = None,
    families_per_domain: int = FAMILIES_PER_DOMAIN,
    stages: set[str] | None = None,
    pilot: bool = False,
    seed: int = 42,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[dict[str, Any]]:
    """Run the synthetic data generation pipeline with composable stages."""
    from httpclient import AsyncClient

    if domains is None:
        domains = list(DOMAIN_TEMPLATES.keys())
    if stages is None:
        stages = set(ALL_STAGES)
    if pilot:
        families_per_domain = min(families_per_domain, 10)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    needs_llm = stages & ({"base"} | VARIANT_STAGES | {"repair", "validate"})

    if needs_llm:
        if not LLM_BASE_URL:
            logger.error("LLM_BASE_URL not set. Set it in .env or environment.")
            return []
        client_headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            client_headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    semaphore = asyncio.Semaphore(max_concurrent)

    families_by_domain: dict[str, list[dict]] = {}
    variants_by_domain: dict[str, list[dict[str, dict | None]]] = {}

    if needs_llm:
        async with AsyncClient(
            headers=client_headers,
            timeout=120,
            pool_size=max_concurrent,
        ) as client:
            # Base generation
            if "base" in stages:
                families_by_domain = await run_base_stage(
                    client, domains, families_per_domain, seed, semaphore
                )
            else:
                for domain in domains:
                    loaded = _load_jsonl(DATA_DIR / f"{domain}_families.jsonl")
                    if loaded:
                        families_by_domain[domain] = loaded
                        logger.info("  Loaded %d families for %s from cache", len(loaded), domain)

            # Variant generation
            requested_variants = stages & VARIANT_STAGES
            if requested_variants and families_by_domain:
                variants_by_domain = await run_variant_stage(
                    client, families_by_domain, requested_variants, semaphore
                )
            else:
                for domain in domains:
                    loaded = _load_jsonl(DATA_DIR / f"{domain}_variants.jsonl")
                    if loaded:
                        variants_by_domain[domain] = loaded

            # Repair bad labels
            if "repair" in stages and variants_by_domain:
                for domain in list(variants_by_domain.keys()):
                    families = families_by_domain.get(domain, [])
                    variants = variants_by_domain[domain]
                    if families and variants:
                        template = DOMAIN_TEMPLATES[domain]
                        updated, repaired, failed = await repair_variants(
                            client, families, variants, template,
                            LLM_REPAIR_MODEL, LLM_BASE_URL,
                            max_concurrent=max_concurrent,
                        )
                        if repaired:
                            variants_by_domain[domain] = updated
                            _save_jsonl(updated, DATA_DIR / f"{domain}_variants.jsonl")

            # Dedup
            if "dedup" in stages and families_by_domain:
                for domain in list(families_by_domain.keys()):
                    families = families_by_domain[domain]
                    kept, dropped = dedup_families(families, threshold=0.7)
                    if dropped:
                        families_by_domain[domain] = kept
                        _save_jsonl(kept, DATA_DIR / f"{domain}_families.jsonl")
                        # Sync variants: remove entries for dropped families
                        variants = variants_by_domain.get(domain, [])
                        if variants:
                            kept_set = set(range(len(families))) - set(dropped)
                            variants_by_domain[domain] = [
                                variants[i] for i in sorted(kept_set) if i < len(variants)
                            ]
                            _save_jsonl(variants_by_domain[domain], DATA_DIR / f"{domain}_variants.jsonl")

            # Validation
            if "validate" in stages and families_by_domain:
                families_by_domain = await run_validate_stage(
                    client, families_by_domain, semaphore
                )
    else:
        for domain in domains:
            loaded = _load_jsonl(DATA_DIR / f"{domain}_families.jsonl")
            if loaded:
                families_by_domain[domain] = loaded
            loaded_v = _load_jsonl(DATA_DIR / f"{domain}_variants.jsonl")
            if loaded_v:
                variants_by_domain[domain] = loaded_v

    # Dedup (works in both LLM and non-LLM paths)
    if "dedup" in stages and families_by_domain and not needs_llm:
        for domain in list(families_by_domain.keys()):
            families = families_by_domain[domain]
            kept, dropped = dedup_families(families, threshold=0.7)
            if dropped:
                families_by_domain[domain] = kept
                _save_jsonl(kept, DATA_DIR / f"{domain}_families.jsonl")
                variants = variants_by_domain.get(domain, [])
                if variants:
                    kept_set = set(range(len(families))) - set(dropped)
                    variants_by_domain[domain] = [
                        variants[i] for i in sorted(kept_set) if i < len(variants)
                    ]
                    _save_jsonl(variants_by_domain[domain], DATA_DIR / f"{domain}_variants.jsonl")

    # Convert to TypedQuestion items
    all_items: list[dict[str, Any]] = []
    for domain, families in families_by_domain.items():
        domain_variants = variants_by_domain.get(domain, [])
        for idx, family in enumerate(families):
            variant_data = domain_variants[idx] if idx < len(domain_variants) else {}
            items = family_to_typed_questions(
                family, variant_data, domain, family.get("_family_idx", idx),
                enabled_stages=stages if stages != set(ALL_STAGES) else None,
            )
            all_items.extend(items)

    # Jev API soft labels
    if "jev-label" in stages and all_items:
        logger.info("Jev API labeling (%d items)...", len(all_items))
        from jev_client import JevClient
        jev = JevClient()
        label_results = label_items_sync(all_items, jev, batch_size=10)
        jev.close()
        label_map = {r.question_id: r for r in label_results}
        for item in all_items:
            lr = label_map.get(item["id"])
            if lr and lr.teacher_probs:
                item["teacher_probs"] = lr.teacher_probs
        print_bucket_report(label_results)

    # Save final output
    out_path = DATA_DIR / "synthetic.jsonl"
    _save_jsonl(all_items, out_path)

    # Print summary
    from collections import Counter
    type_counts = Counter(item["question"]["type"] for item in all_items)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total items:  %d", len(all_items))
    logger.info("  By type:      noul=%d  choice=%d  score=%d", type_counts["noul"], type_counts["choice"], type_counts["score"])
    logger.info("  Output:       %s", out_path)

    return all_items


EXPECTED_PER_FAMILY = 35  # 9 base + 6 cf + 9 para-state + 4 para-q + 4 neg + ~3 shuffle


def run_report(domains: list[str]) -> None:
    """Report generation status for each domain from cached files."""
    from collections import Counter

    logger.info("=" * 72)
    logger.info(
        "%-25s %8s %8s %8s %10s %10s %6s",
        "Domain", "Families", "Variants", "Items", "Per-family", "Expected", "Gap",
    )
    logger.info("-" * 72)

    total_families = 0
    total_items = 0
    all_types = Counter()

    for domain in domains:
        families = _load_jsonl(DATA_DIR / f"{domain}_families.jsonl")
        variants = _load_jsonl(DATA_DIR / f"{domain}_variants.jsonl")
        n_fam = len(families)
        n_var = len(variants)

        items: list[dict] = []
        for idx, fam in enumerate(families):
            vd = variants[idx] if idx < len(variants) else {}
            items.extend(family_to_typed_questions(fam, vd, domain, fam.get("_family_idx", idx)))

        n_items = len(items)
        per_fam = n_items / n_fam if n_fam else 0
        expected = EXPECTED_PER_FAMILY * n_fam
        gap_pct = ((n_items - expected) / expected * 100) if expected else 0

        types = Counter(i["question"]["type"] for i in items)
        all_types += types

        # Variant breakdown
        v_counts = Counter()
        for i in items:
            iid = i["id"]
            if "-cf-" in iid:
                v_counts["cf"] += 1
            elif "-para-state-" in iid:
                v_counts["para-state"] += 1
            elif "-para-q-" in iid:
                v_counts["para-q"] += 1
            elif "-neg-" in iid:
                v_counts["neg"] += 1
            elif "-shuffle-" in iid:
                v_counts["shuffle"] += 1
            else:
                v_counts["base"] += 1

        # Bad labels in variants
        bad_cf_choice = 0
        bad_cf_score = 0
        template = DOMAIN_TEMPLATES.get(domain, {})
        for idx, (fam, var) in enumerate(zip(families, variants)):
            cf = var.get("counterfactual", {})
            if cf:
                choice_qs = fam.get("choice_questions", [])
                if not choice_qs and fam.get("choice_question"):
                    choice_qs = [fam["choice_question"]]
                if choice_qs and cf.get("choice_label"):
                    criteria = choice_qs[0].get("criteria", template.get("choice_template", {}).get("criteria", {}))
                    if cf["choice_label"] not in criteria:
                        bad_cf_choice += 1
                if cf.get("score_label") is not None:
                    score_qs = fam.get("score_questions", [])
                    if not score_qs and fam.get("score_question"):
                        score_qs = [fam["score_question"]]
                    if score_qs:
                        criteria = score_qs[0].get("criteria", template.get("score_template", {}).get("criteria", []))
                        if not (1.0 <= float(cf["score_label"]) <= float(len(criteria))):
                            bad_cf_score += 1

        logger.info(
            "%-25s %8d %8d %8d %10.1f %10d %+5.0f%%",
            domain, n_fam, n_var, n_items, per_fam, expected, gap_pct,
        )
        logger.info(
            "  types: noul=%-5d choice=%-5d score=%-5d",
            types["noul"], types["choice"], types["score"],
        )
        logger.info(
            "  variants: base=%-5d cf=%-5d para-s=%-5d para-q=%-5d neg=%-5d shuffle=%-5d",
            v_counts["base"], v_counts["cf"], v_counts["para-state"],
            v_counts["para-q"], v_counts["neg"], v_counts["shuffle"],
        )
        if bad_cf_choice or bad_cf_score:
            logger.info(
                "  bad labels: cf-choice=%d cf-score=%d (run --stages repair to fix)",
                bad_cf_choice, bad_cf_score,
            )

        total_families += n_fam
        total_items += n_items

    logger.info("-" * 72)
    logger.info(
        "%-25s %8d %8s %8d %10.1f %10d %+5.0f%%",
        "TOTAL", total_families, "",  total_items,
        total_items / total_families if total_families else 0,
        EXPECTED_PER_FAMILY * total_families,
        ((total_items - EXPECTED_PER_FAMILY * total_families) / (EXPECTED_PER_FAMILY * total_families) * 100) if total_families else 0,
    )
    logger.info(
        "  types: noul=%-5d choice=%-5d score=%-5d",
        all_types["noul"], all_types["choice"], all_types["score"],
    )


def push_to_hf(repo_id: str, path: Path | None = None) -> None:
    """Push synthetic data to a HuggingFace dataset repository."""
    import subprocess

    if path is None:
        path = DATA_DIR / "synthetic.jsonl"

    if not path.exists():
        logger.error("%s does not exist. Generate data first.", path)
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="synthetic.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update synthetic data ({path.stat().st_size // 1024}KB)",
        )
        logger.info("Pushed %s to https://huggingface.co/datasets/%s", path, repo_id)
    except ImportError:
        result = subprocess.run(
            ["huggingface-cli", "upload", "--repo-type", "dataset",
             repo_id, str(path), "synthetic.jsonl"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Pushed %s to https://huggingface.co/datasets/%s", path, repo_id)
        else:
            logger.error("HF upload failed: %s", result.stderr)
            logger.error("Install huggingface_hub: pip install huggingface_hub")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic typed-decision training data"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        choices=list(DOMAIN_TEMPLATES.keys()),
        help="Domains to generate (default: all)",
    )
    parser.add_argument(
        "--families-per-domain",
        type=int,
        default=FAMILIES_PER_DOMAIN,
        help=f"Families per domain (default: {FAMILIES_PER_DOMAIN})",
    )
    parser.add_argument(
        "--stages",
        type=lambda s: set(s.split(",")),
        default=None,
        help=f"Comma-separated stages to run (default: all). Available: {','.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: max 10 families per domain",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent API requests (default: {MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--push-to-hf",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Push output to HuggingFace dataset repo (e.g., Oaklight/jev-synthetic)",
    )
    # Legacy compat flags
    parser.add_argument("--skip-validation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-jev-labels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-variants", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Handle legacy flags
    stages = args.stages
    if stages is not None:
        invalid = stages - set(ALL_STAGES)
        if invalid:
            parser.error(f"Unknown stages: {', '.join(sorted(invalid))}. Available: {', '.join(ALL_STAGES)}")
    if stages is None:
        stages = set(ALL_STAGES)
        if args.skip_validation:
            stages.discard("validate")
        if args.skip_jev_labels:
            stages.discard("jev-label")
        if args.skip_variants:
            stages -= VARIANT_STAGES
            stages.discard("shuffle")

    pipeline_stages = stages - {"report"}
    if pipeline_stages:
        asyncio.run(
            run_pipeline(
                domains=args.domains,
                families_per_domain=args.families_per_domain,
                stages=pipeline_stages,
                pilot=args.pilot,
                seed=args.seed,
                max_concurrent=args.max_concurrent,
            )
        )

    if "report" in stages:
        run_report(args.domains or list(DOMAIN_TEMPLATES.keys()))

    if args.push_to_hf:
        push_to_hf(args.push_to_hf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
