"""Label noul items using LLM-generated confounders via Jev API.

Similar to label_augmented_noul.py but uses LLM confounders
(augmented_options_llm) instead of template-based augmented_options.

Usage:
    python data/scripts/label_confounder_noul_jev.py                  # dry-run
    python data/scripts/label_confounder_noul_jev.py --apply          # label all
    python data/scripts/label_confounder_noul_jev.py --apply --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "_vendor"))
sys.path.insert(0, str(_repo_root / "probing" / "scripts"))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env")

sys.path.insert(0, str(_repo_root))
from data.format import TypedQuestion
from data.synthetic_label import label_items_sync

logger = logging.getLogger("label_confounder_noul_jev")

DATA_DIR = _repo_root / "data" / "benchmarks" / "synthetic"
LABEL_KEY = "jev_augmented_llm"


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


def assemble_confounder_noul(domains: list[str]) -> dict[str, list[dict]]:
    """Build TypedQuestion dicts using augmented_options_llm from families."""
    items_by_domain: dict[str, list[dict]] = {}
    for domain in domains:
        fam_path = DATA_DIR / f"{domain}_families.jsonl"
        if not fam_path.exists():
            continue
        families = _load_jsonl(fam_path)
        items = []
        for fam_idx, fam in enumerate(families):
            fidx = fam.get("_family_idx", fam_idx)
            for nq_idx, nq in enumerate(fam.get("noul_questions", [])):
                llm_options = nq.get("augmented_options_llm")
                if not llm_options:
                    continue
                item_id = f"synthetic-{domain}-{fidx:04d}-noul-{nq.get('cognitive_type', 'unknown')}-{nq_idx}-llm"
                item = TypedQuestion.noul(
                    id=item_id,
                    state=fam.get("state", ""),
                    instructions=nq.get("instructions", ""),
                    label=nq.get("label", True),
                    source=f"synthetic-{domain}",
                    split="train",
                    augmented_options=llm_options,
                ).to_dict()
                items.append(item)
        if items:
            items_by_domain[domain] = items
    return items_by_domain


def find_unlabeled(items_by_domain: dict[str, list[dict]]) -> dict[str, list[dict]]:
    unlabeled: dict[str, list[dict]] = {}
    for domain, items in items_by_domain.items():
        label_path = DATA_DIR / f"{domain}_{LABEL_KEY}_labels.jsonl"
        existing_ids = {lbl["id"] for lbl in _load_jsonl(label_path)}
        missing = [it for it in items if it["id"] not in existing_ids]
        if missing:
            unlabeled[domain] = missing
    return unlabeled


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Label noul items via Jev API using LLM-generated confounders.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--domain", type=str)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    if args.domain:
        domains = [args.domain]
    else:
        domains = _discover_domains()
        if args.shard is not None:
            domains = [d for i, d in enumerate(sorted(domains)) if i % args.num_shards == args.shard]
            logger.info("Shard %d/%d: %d domains", args.shard, args.num_shards, len(domains))

    all_items = assemble_confounder_noul(domains)
    total = sum(len(v) for v in all_items.values())
    logger.info("Found %d confounder noul items across %d domains", total, len(all_items))

    unlabeled = find_unlabeled(all_items)
    total_unlabeled = sum(len(v) for v in unlabeled.values())
    logger.info("Unlabeled: %d items across %d domains", total_unlabeled, len(unlabeled))

    if not unlabeled:
        logger.info("All confounder noul items already labeled.")
        return

    logger.info("")
    logger.info("%-35s %8s %8s", "Domain", "Total", "Unlabeled")
    logger.info("-" * 55)
    for domain in sorted(unlabeled.keys()):
        logger.info("  %-33s %8d %8d", domain, len(all_items.get(domain, [])), len(unlabeled[domain]))
    logger.info("-" * 55)
    logger.info("  %-33s %8d %8d", "TOTAL", total, total_unlabeled)

    if not args.apply:
        logger.info("")
        logger.info("Dry-run mode. Use --apply to label via Jev API.")
        return

    from jev_client import JevClient

    jev = JevClient()
    total_labeled = 0

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
                new_labels.append({
                    "id": r.question_id,
                    "teacher_probs": {LABEL_KEY: r.teacher_probs},
                })

        if new_labels:
            label_path = DATA_DIR / f"{domain}_{LABEL_KEY}_labels.jsonl"
            _append_jsonl(new_labels, label_path)
            total_labeled += len(new_labels)
            logger.info(
                "  %s: %d/%d labeled (%.1fs), saved to %s",
                domain, len(new_labels), len(items), elapsed, label_path.name,
            )
        else:
            logger.warning("  %s: no labels produced for %d items", domain, len(items))

    jev.close()

    logger.info("")
    logger.info("DONE: %d / %d labeled", total_labeled, total_unlabeled)

    remaining = find_unlabeled(all_items)
    remaining_count = sum(len(v) for v in remaining.values())
    if remaining_count:
        logger.info("Still unlabeled: %d items", remaining_count)
    else:
        logger.info("All confounder noul items labeled!")


if __name__ == "__main__":
    main()
