"""Fill missing GPT-5.6 Luna teacher labels for synthetic data items.

Identifies items that have Jev labels but no Luna labels, then calls
the LLM labeling pipeline to fill the gaps. Supports sharding for
parallel execution and arbitrary model/endpoint override.

Usage:
    python data/scripts/fill_luna_gaps.py              # dry-run: report gaps only
    python data/scripts/fill_luna_gaps.py --apply       # call Luna and fill gaps
    python data/scripts/fill_luna_gaps.py --domain game_strategy --apply  # one domain
    python data/scripts/fill_luna_gaps.py --apply --model Qwen/Qwen3.8-27B --base-url http://rbdgx3:8765
    python data/scripts/fill_luna_gaps.py --apply --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "_vendor"))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env")

# Imports from the data package — run as: python data/scripts/fill_luna_gaps.py
# so we need the repo root on sys.path for the `data` package.
sys.path.insert(0, str(_repo_root))
from data.format import TypedQuestion
from data.synthetic import family_to_typed_questions
from data.synthetic_llm_label import label_items_llm

logger = logging.getLogger("fill_luna_gaps")

DATA_DIR = _repo_root / "data" / "benchmarks" / "synthetic"
TEACHER_KEY = "gpt_5_6_luna"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning an empty list if it does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl_batch(items: list[dict], path: Path) -> None:
    """Append multiple records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_domain(item_id: str) -> str:
    """Extract domain name from a synthetic item ID.

    IDs look like: synthetic-{domain}-{NNNN}-{type}-{subtype}
    Domains use underscores, so split on '-' index 1 is safe.
    """
    return item_id.split("-")[1]


def _classify_variant(item_id: str) -> str:
    """Classify an item ID into its variant type.

    Returns one of: base, cf, para-state, para-q, shuffle, neg.
    """
    # Remove the synthetic-{domain}-{NNNN}- prefix
    parts = item_id.split("-")
    # parts[0] = "synthetic", parts[1] = domain, parts[2] = family_idx
    remainder = "-".join(parts[3:])
    if remainder.startswith("cf-"):
        return "cf"
    if remainder.startswith("para-state-"):
        return "para-state"
    if remainder.startswith("para-q-"):
        return "para-q"
    if remainder.startswith("shuffle-"):
        return "shuffle"
    if remainder.startswith("neg-"):
        return "neg"
    return "base"


def _classify_qtype(item_id: str) -> str:
    """Extract question type (noul/choice/score) from item ID."""
    # After variant prefix, the type marker is the next segment
    parts = item_id.split("-")
    remainder = "-".join(parts[3:])
    # Strip variant prefix
    for prefix in ("cf-", "para-state-", "para-q-", "shuffle-", "neg-"):
        if remainder.startswith(prefix):
            remainder = remainder[len(prefix):]
            break
    # Now remainder starts with noul/choice/score
    if remainder.startswith("noul"):
        return "noul"
    if remainder.startswith("choice"):
        return "choice"
    if remainder.startswith("score"):
        return "score"
    return "unknown"


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

def find_gaps(
    domains: list[str] | None = None,
) -> dict[str, set[str]]:
    """Find item IDs that have Jev labels but no Luna labels.

    Args:
        domains: If given, only check these domains.

    Returns:
        Dict mapping domain -> set of missing item IDs.
    """
    all_domains = _discover_domains()
    if domains:
        all_domains = [d for d in all_domains if d in domains]

    gaps: dict[str, set[str]] = {}
    for domain in all_domains:
        jev_path = DATA_DIR / f"{domain}_jev_labels.jsonl"
        luna_path = DATA_DIR / f"{domain}_{TEACHER_KEY}_labels.jsonl"

        jev_ids = {r["id"] for r in _load_jsonl(jev_path)}
        luna_ids = {r["id"] for r in _load_jsonl(luna_path)}

        missing = jev_ids - luna_ids
        if missing:
            gaps[domain] = missing

    return gaps


def _discover_domains() -> list[str]:
    """Discover all domains that have Jev label files."""
    domains = []
    for path in sorted(DATA_DIR.glob("*_jev_labels.jsonl")):
        domain = path.name.replace("_jev_labels.jsonl", "")
        domains.append(domain)
    return domains


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_gap_report(gaps: dict[str, set[str]]) -> None:
    """Print a summary of gaps per domain, question type, and variant."""
    total = sum(len(ids) for ids in gaps.values())
    if total == 0:
        logger.info("No gaps found — all Jev-labeled items have Luna labels.")
        return

    logger.info("=" * 70)
    logger.info("Luna label gaps: %d items across %d domains", total, len(gaps))
    logger.info("=" * 70)

    # Per-domain summary
    logger.info("")
    logger.info("%-30s %6s  %s", "Domain", "Gaps", "By type (noul/choice/score)")
    logger.info("-" * 70)

    for domain in sorted(gaps, key=lambda d: -len(gaps[d])):
        ids = gaps[domain]
        by_type: dict[str, int] = defaultdict(int)
        for item_id in ids:
            by_type[_classify_qtype(item_id)] += 1
        type_str = ", ".join(f"{t}={c}" for t, c in sorted(by_type.items()))
        logger.info("%-30s %6d  %s", domain, len(ids), type_str)

    # Per-variant summary
    logger.info("")
    logger.info("By variant:")
    variant_counts: dict[str, int] = defaultdict(int)
    for ids in gaps.values():
        for item_id in ids:
            variant_counts[_classify_variant(item_id)] += 1
    for variant in sorted(variant_counts, key=lambda v: -variant_counts[v]):
        logger.info("  %-20s %6d", variant, variant_counts[variant])

    logger.info("")
    logger.info("Total: %d items missing Luna labels", total)


# ---------------------------------------------------------------------------
# Item assembly
# ---------------------------------------------------------------------------

def assemble_items_for_labeling(
    gaps: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Load families/variants and convert to TypedQuestion dicts for missing items.

    Returns:
        Dict mapping domain -> list of item dicts ready for label_items_llm().
    """
    items_by_domain: dict[str, list[dict[str, Any]]] = {}

    for domain, missing_ids in sorted(gaps.items()):
        families_path = DATA_DIR / f"{domain}_families.jsonl"
        variants_path = DATA_DIR / f"{domain}_variants.jsonl"

        if not families_path.exists():
            logger.warning(
                "Skipping %s: families file not found at %s", domain, families_path
            )
            continue

        families = _load_jsonl(families_path)
        variants_list = _load_jsonl(variants_path) if variants_path.exists() else []

        domain_items: list[dict[str, Any]] = []
        for idx, family in enumerate(families):
            family_idx = family.get("_family_idx", idx)
            variant_data = variants_list[idx] if idx < len(variants_list) else {}
            all_family_items = family_to_typed_questions(
                family, variant_data, domain, family_idx
            )
            # Filter to only the items with missing Luna labels
            for item in all_family_items:
                if item["id"] in missing_ids:
                    domain_items.append(item)

        if domain_items:
            items_by_domain[domain] = domain_items
            logger.info(
                "  %s: assembled %d/%d missing items from %d families",
                domain, len(domain_items), len(missing_ids), len(families),
            )
            # Warn about IDs we couldn't reconstruct
            reconstructed = {it["id"] for it in domain_items}
            unreconstructed = missing_ids - reconstructed
            if unreconstructed:
                logger.warning(
                    "  %s: %d missing IDs could not be reconstructed from families",
                    domain, len(unreconstructed),
                )

    return items_by_domain


# ---------------------------------------------------------------------------
# Labeling and merging
# ---------------------------------------------------------------------------

async def fill_gaps(
    gaps: dict[str, set[str]],
    max_concurrent: int = 10,
    model_override: str | None = None,
    base_url_override: str | None = None,
) -> None:
    """Call LLM to label missing items and append results to label files."""
    base_url = base_url_override or os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "") if not base_url_override else ""
    teacher_model = model_override or os.environ.get("LLM_TEACHER_MODEL", "argo:gpt-5.6-luna")

    if not base_url:
        logger.error("LLM_BASE_URL not set — cannot call Luna for labeling")
        return

    items_by_domain = assemble_items_for_labeling(gaps)
    if not items_by_domain:
        logger.info("No items to label after assembly.")
        return

    total_items = sum(len(v) for v in items_by_domain.values())
    logger.info(
        "Labeling %d items across %d domains with %s ...",
        total_items, len(items_by_domain), teacher_model,
    )

    # Import the vendored async HTTP client
    from httpclient import AsyncClient

    client_headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        client_headers["Authorization"] = f"Bearer {api_key}"

    async with AsyncClient(
        headers=client_headers, timeout=120, pool_size=max_concurrent
    ) as client:
        for domain, items in sorted(items_by_domain.items()):
            logger.info("  %s: labeling %d items ...", domain, len(items))

            label_results = await label_items_llm(
                client, items, teacher_model, base_url,
                max_concurrent=max_concurrent,
            )

            # Build new label records
            new_labels = []
            for r in label_results:
                if r.teacher_probs:
                    new_labels.append({
                        "id": r.item_id,
                        "teacher_probs": {TEACHER_KEY: r.teacher_probs},
                    })

            if new_labels:
                label_path = DATA_DIR / f"{domain}_{TEACHER_KEY}_labels.jsonl"
                _append_jsonl_batch(new_labels, label_path)
                logger.info(
                    "  %s: appended %d new labels (of %d attempted) to %s",
                    domain, len(new_labels), len(items), label_path.name,
                )
            else:
                logger.warning("  %s: no labels produced for %d items", domain, len(items))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify and fill missing Luna teacher labels.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually call LLM and fill gaps (default: dry-run report only)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Only process a specific domain (e.g. game_strategy)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent LLM requests (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model (e.g. Qwen/Qwen3.8-27B)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override LLM base URL (e.g. http://rbdgx3:8765)",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard index (0-based) for parallel execution",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    domains = [args.domain] if args.domain else None

    # Step 1: Find gaps
    logger.info("Scanning for Luna label gaps ...")
    gaps = find_gaps(domains=domains)

    # Apply sharding if requested
    if args.shard is not None and args.num_shards is not None:
        all_domains = sorted(gaps.keys())
        shard_domains = [d for i, d in enumerate(all_domains) if i % args.num_shards == args.shard]
        gaps = {d: gaps[d] for d in shard_domains if d in gaps}
        logger.info("Shard %d/%d: processing %d domains: %s",
                     args.shard, args.num_shards, len(shard_domains),
                     ", ".join(shard_domains))

    print_gap_report(gaps)

    if not gaps:
        return

    if not args.apply:
        logger.info("")
        logger.info("Dry-run mode. Use --apply to fill gaps.")
        return

    # Step 2: Fill gaps
    asyncio.run(fill_gaps(
        gaps,
        max_concurrent=args.max_concurrent,
        model_override=args.model,
        base_url_override=args.base_url,
    ))

    # Step 3: Verify remaining gaps
    logger.info("")
    logger.info("Verifying remaining gaps ...")
    remaining = find_gaps(domains=domains)
    remaining_total = sum(len(ids) for ids in remaining.values())
    if remaining_total == 0:
        logger.info("All gaps filled successfully.")
    else:
        logger.info("%d items still missing (likely unreconstructable from families).", remaining_total)


if __name__ == "__main__":
    main()
