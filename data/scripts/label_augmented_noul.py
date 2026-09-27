"""Label augmented noul items via Jev API as choice questions.

Reads assembled augmented noul items from synthetic data, sends them to
Jev's choice API to get 5-option probability distributions, and saves
results to per-domain label files. Resumable and shardable.

Usage:
    python data/scripts/label_augmented_noul.py                    # dry-run
    python data/scripts/label_augmented_noul.py --apply            # label all
    python data/scripts/label_augmented_noul.py --apply --domain medical_triage
    python data/scripts/label_augmented_noul.py --apply --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
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
from data.synthetic_label import label_items_sync

logger = logging.getLogger("label_augmented_noul")

DATA_DIR = _repo_root / "data" / "benchmarks" / "synthetic"
LABEL_KEY = "jev_augmented"


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


def assemble_augmented_noul(domains: list[str]) -> dict[str, list[dict]]:
    """Assemble augmented noul items per domain."""
    items_by_domain: dict[str, list[dict]] = {}

    for domain in domains:
        fam_path = DATA_DIR / f"{domain}_families.jsonl"
        var_path = DATA_DIR / f"{domain}_variants.jsonl"
        if not fam_path.exists():
            continue

        families = _load_jsonl(fam_path)
        variants = _load_jsonl(var_path) if var_path.exists() else []

        items = []
        for idx, fam in enumerate(families):
            vd = variants[idx] if idx < len(variants) else {}
            for it in family_to_typed_questions(fam, vd, domain, idx, enabled_stages=None):
                if (
                    it.get("question", {}).get("type") == "noul"
                    and "augmented_options" in it.get("question", {})
                ):
                    items.append(it)

        if items:
            items_by_domain[domain] = items

    return items_by_domain


def find_unlabeled(
    items_by_domain: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Filter to items not yet labeled with LABEL_KEY."""
    unlabeled: dict[str, list[dict]] = {}

    for domain, items in items_by_domain.items():
        label_path = DATA_DIR / f"{domain}_{LABEL_KEY}_labels.jsonl"
        existing_ids = set()
        if label_path.exists():
            for lbl in _load_jsonl(label_path):
                existing_ids.add(lbl["id"])

        missing = [it for it in items if it["id"] not in existing_ids]
        if missing:
            unlabeled[domain] = missing

    return unlabeled


def shard_domains(domains: list[str], shard: int, num_shards: int) -> list[str]:
    """Select domains for this shard (round-robin by sorted order)."""
    return [d for i, d in enumerate(sorted(domains)) if i % num_shards == shard]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Label augmented noul items via Jev API.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually call Jev API.")
    parser.add_argument("--domain", type=str, help="Label a single domain.")
    parser.add_argument(
        "--shard", type=int, default=None, help="Shard index (0-based)."
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards."
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Jev API batch size."
    )
    args = parser.parse_args()

    if args.domain:
        domains = [args.domain]
    else:
        domains = _discover_domains()
        if args.shard is not None:
            domains = shard_domains(domains, args.shard, args.num_shards)
            logger.info(
                "Shard %d/%d: %d domains", args.shard, args.num_shards, len(domains)
            )

    logger.info("Assembling augmented noul items...")
    all_items = assemble_augmented_noul(domains)
    total = sum(len(v) for v in all_items.values())
    logger.info("Found %d augmented noul items across %d domains", total, len(all_items))

    unlabeled = find_unlabeled(all_items)
    total_unlabeled = sum(len(v) for v in unlabeled.values())
    logger.info("Unlabeled: %d items across %d domains", total_unlabeled, len(unlabeled))

    if not unlabeled:
        logger.info("All augmented noul items already labeled.")
        return

    logger.info("")
    logger.info("%-35s %8s %8s", "Domain", "Total", "Unlabeled")
    logger.info("-" * 55)
    for domain in sorted(unlabeled.keys()):
        n_total = len(all_items.get(domain, []))
        n_unlabeled = len(unlabeled[domain])
        logger.info("  %-33s %8d %8d", domain, n_total, n_unlabeled)
    logger.info("-" * 55)
    logger.info("  %-33s %8d %8d", "TOTAL", total, total_unlabeled)

    if not args.apply:
        logger.info("")
        logger.info("Dry-run mode. Use --apply to label via Jev API.")
        return

    # Label via Jev API
    from jev_client import JevClient

    jev = JevClient()
    total_labeled = 0
    total_failed = 0

    for domain in sorted(unlabeled.keys()):
        items = unlabeled[domain]
        logger.info("")
        logger.info("  %s: labeling %d items...", domain, len(items))

        t0 = time.monotonic()
        results = label_items_sync(items, jev, batch_size=args.batch_size)
        elapsed = time.monotonic() - t0

        new_labels = []
        for r in results:
            if r.teacher_probs and len(r.teacher_probs) > 2:
                new_labels.append(
                    {
                        "id": r.question_id,
                        "teacher_probs": {LABEL_KEY: r.teacher_probs},
                    }
                )

        if new_labels:
            label_path = DATA_DIR / f"{domain}_{LABEL_KEY}_labels.jsonl"
            _append_jsonl(new_labels, label_path)
            total_labeled += len(new_labels)
            logger.info(
                "  %s: %d/%d labeled (%.1fs), saved to %s",
                domain,
                len(new_labels),
                len(items),
                elapsed,
                label_path.name,
            )
        else:
            total_failed += len(items)
            logger.warning("  %s: no labels produced for %d items", domain, len(items))

    jev.close()

    logger.info("")
    logger.info("=" * 55)
    logger.info("DONE: %d labeled, %d failed", total_labeled, total_failed)

    # Verify
    remaining = find_unlabeled(all_items)
    remaining_count = sum(len(v) for v in remaining.values())
    if remaining_count:
        logger.info("Still unlabeled: %d items", remaining_count)
    else:
        logger.info("All augmented noul items labeled!")


if __name__ == "__main__":
    main()
