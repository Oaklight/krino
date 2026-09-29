"""Label LLM-generated confounders via Jev API and/or LLM teacher.

Reads augmented noul items from families, swaps augmented_options_llm into
augmented_options so existing labeling code routes them as choice questions,
then generates soft labels. Outputs to separate label files:
  {domain}_jev_augmented_llm_labels.jsonl
  {domain}_gpt_5_6_luna_augmented_llm_labels.jsonl

Usage:
    python data/scripts/label_llm_confounders.py                          # dry-run
    python data/scripts/label_llm_confounders.py --apply --teacher jev
    python data/scripts/label_llm_confounders.py --apply --teacher luna
    python data/scripts/label_llm_confounders.py --apply --teacher jev --domain medical_triage
    python data/scripts/label_llm_confounders.py --apply --teacher jev --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "_vendor"))
sys.path.insert(0, str(_repo_root / "probing" / "scripts"))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env")

sys.path.insert(0, str(_repo_root))
from data.synthetic import family_to_typed_questions

logger = logging.getLogger("label_llm_confounders")

DATA_DIR = _repo_root / "data" / "benchmarks" / "synthetic"

TEACHER_CONFIGS = {
    "jev": {
        "label_key": "jev_augmented_llm",
    },
    "luna": {
        "label_key": "gpt_5_6_luna_augmented_llm",
        "model": "argo:gpt-5.6-luna",
    },
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(items: list[dict], path: Path) -> None:
    with open(path, "a") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _discover_domains() -> list[str]:
    return sorted(
        f.name.replace("_families.jsonl", "")
        for f in DATA_DIR.glob("*_families.jsonl")
    )


def shard_domains(domains: list[str], shard: int, num_shards: int) -> list[str]:
    return [d for i, d in enumerate(sorted(domains)) if i % num_shards == shard]


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probability distribution to sum to 1.0."""
    total = sum(probs.values())
    if total <= 0:
        return probs
    if abs(total - 1.0) < 1e-6:
        return probs
    return {k: v / total for k, v in probs.items()}


def assemble_llm_confounder_items(domains: list[str]) -> dict[str, list[dict]]:
    """Assemble noul items with LLM confounders swapped into augmented_options.

    family_to_typed_questions only propagates augmented_options (template),
    not augmented_options_llm. We build a lookup from the family's noul
    questions keyed by cognitive_type, then patch each emitted item.
    """
    items_by_domain: dict[str, list[dict]] = {}

    for domain in domains:
        fam_path = DATA_DIR / f"{domain}_families.jsonl"
        var_path = DATA_DIR / f"{domain}_variants.jsonl"
        if not fam_path.exists():
            continue

        families = _load_jsonl(fam_path)
        variants = _load_jsonl(var_path) if var_path.exists() else []

        # Build per-family lookup: cognitive_type → augmented_options_llm
        llm_opts_by_family: list[dict[str, dict]] = []
        for fam in families:
            ct_map = {}
            for nq in fam.get("noul_questions", []):
                opts = nq.get("augmented_options_llm")
                if opts:
                    ct_map[nq.get("cognitive_type", "")] = opts
            llm_opts_by_family.append(ct_map)

        items = []
        for idx, fam in enumerate(families):
            vd = variants[idx] if idx < len(variants) else {}
            ct_map = llm_opts_by_family[idx]
            if not ct_map:
                continue

            for it in family_to_typed_questions(fam, vd, domain, idx, enabled_stages=None):
                q = it.get("question", {})
                if q.get("type") != "noul" or "augmented_options" not in q:
                    continue
                # Match by cognitive_type suffix in the item ID
                item_id = it.get("id", "")
                ct = item_id.rsplit("-noul-", 1)[-1] if "-noul-" in item_id else ""
                llm_opts = ct_map.get(ct)
                if llm_opts:
                    q["augmented_options"] = llm_opts
                    items.append(it)

        if items:
            items_by_domain[domain] = items

    return items_by_domain


def find_unlabeled(
    items_by_domain: dict[str, list[dict]],
    label_key: str,
) -> dict[str, list[dict]]:
    unlabeled: dict[str, list[dict]] = {}
    for domain, items in items_by_domain.items():
        label_path = DATA_DIR / f"{domain}_{label_key}_labels.jsonl"
        existing_ids = set()
        if label_path.exists():
            for lbl in _load_jsonl(label_path):
                existing_ids.add(lbl["id"])
        missing = [it for it in items if it["id"] not in existing_ids]
        if missing:
            unlabeled[domain] = missing
    return unlabeled


def label_jev(
    unlabeled: dict[str, list[dict]],
    label_key: str,
    batch_size: int,
) -> None:
    from data.synthetic_label import label_items_sync
    from jev_client import JevClient

    jev = JevClient()
    total_labeled = 0
    total_failed = 0

    for domain in sorted(unlabeled.keys()):
        items = unlabeled[domain]
        logger.info("  %s: labeling %d items via Jev...", domain, len(items))

        t0 = time.monotonic()
        results = label_items_sync(items, jev, batch_size=batch_size)
        elapsed = time.monotonic() - t0

        new_labels = []
        for r in results:
            if r.teacher_probs and len(r.teacher_probs) > 2:
                probs = _normalize_probs(r.teacher_probs)
                new_labels.append(
                    {"id": r.question_id, "teacher_probs": {label_key: probs}}
                )

        if new_labels:
            label_path = DATA_DIR / f"{domain}_{label_key}_labels.jsonl"
            _append_jsonl(new_labels, label_path)
            total_labeled += len(new_labels)
            logger.info(
                "  %s: %d/%d labeled (%.1fs)",
                domain, len(new_labels), len(items), elapsed,
            )
        else:
            total_failed += len(items)
            logger.warning("  %s: no labels produced for %d items", domain, len(items))

    jev.close()
    logger.info("Jev total: %d labeled, %d failed", total_labeled, total_failed)


async def label_luna(
    unlabeled: dict[str, list[dict]],
    label_key: str,
    model: str,
    max_concurrent: int,
) -> None:
    from data.synthetic_llm_label import label_items_llm

    sys.path.insert(0, str(_repo_root / "_vendor"))
    from httpclient import AsyncClient

    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    total_labeled = 0
    total_failed = 0

    async with AsyncClient(headers=headers, timeout=120, pool_size=max_concurrent) as client:
        for domain in sorted(unlabeled.keys()):
            items = unlabeled[domain]
            logger.info("  %s: labeling %d items via %s...", domain, len(items), model)

            t0 = time.monotonic()
            results = await label_items_llm(
                client, items, model, base_url, max_concurrent=max_concurrent,
            )
            elapsed = time.monotonic() - t0

            new_labels = []
            for r in results:
                if r.teacher_probs:
                    probs = _normalize_probs(r.teacher_probs)
                    new_labels.append(
                        {"id": r.item_id, "teacher_probs": {label_key: probs}}
                    )

            if new_labels:
                label_path = DATA_DIR / f"{domain}_{label_key}_labels.jsonl"
                _append_jsonl(new_labels, label_path)
                total_labeled += len(new_labels)
                logger.info(
                    "  %s: %d/%d labeled (%.1fs)",
                    domain, len(new_labels), len(items), elapsed,
                )
            else:
                total_failed += len(items)
                logger.warning(
                    "  %s: no labels produced for %d items", domain, len(items),
                )

    logger.info("Luna total: %d labeled, %d failed", total_labeled, total_failed)


async def main_async(args: argparse.Namespace) -> None:
    teacher = args.teacher
    config = TEACHER_CONFIGS[teacher]
    label_key = config["label_key"]

    if args.domain:
        domains = [args.domain]
    else:
        domains = _discover_domains()
        if args.shard is not None:
            domains = shard_domains(domains, args.shard, args.num_shards)
            logger.info("Shard %d/%d: %d domains", args.shard, args.num_shards, len(domains))

    logger.info("Assembling LLM confounder items...")
    all_items = assemble_llm_confounder_items(domains)
    total = sum(len(v) for v in all_items.values())
    logger.info("Found %d LLM confounder items across %d domains", total, len(all_items))

    unlabeled = find_unlabeled(all_items, label_key)
    total_unlabeled = sum(len(v) for v in unlabeled.values())
    logger.info("Unlabeled (%s): %d items", label_key, total_unlabeled)

    if not total_unlabeled:
        logger.info("All items already labeled with %s.", label_key)
        return

    if not args.apply:
        logger.info("")
        logger.info("%-35s %8s %8s", "Domain", "Total", "Unlabeled")
        logger.info("-" * 55)
        for domain in sorted(unlabeled.keys()):
            n_total = len(all_items.get(domain, []))
            n_unlabeled = len(unlabeled[domain])
            logger.info("  %-33s %8d %8d", domain, n_total, n_unlabeled)
        logger.info("-" * 55)
        logger.info("  %-33s %8d %8d", "TOTAL", total, total_unlabeled)
        logger.info("\nDry-run. Use --apply to label.")
        return

    if teacher == "jev":
        label_jev(unlabeled, label_key, args.batch_size)
    elif teacher == "luna":
        await label_luna(
            unlabeled, label_key, config["model"], args.max_concurrent,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Label LLM confounders via Jev/Luna.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--teacher", choices=["jev", "luna"], required=True)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
