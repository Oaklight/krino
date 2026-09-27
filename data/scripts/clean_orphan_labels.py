"""Remove orphan label entries whose IDs are no longer produced by the pipeline.

Validation logic in `family_to_typed_questions` has grown stricter over time
(e.g. dropping malformed choice questions), so some previously-generated
label entries in `*_jev_labels.jsonl` / `*_gpt_5_6_luna_labels.jsonl` now
reference item IDs that the current code no longer emits. This script
recomputes the set of valid item IDs per domain and drops any label entries
that don't match, overwriting the label files in place.

Usage:
    python data/scripts/clean_orphan_labels.py            # apply and report
    python data/scripts/clean_orphan_labels.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from data.synthetic import family_to_typed_questions

logger = logging.getLogger("clean_orphan_labels")

DATA_DIR = _repo_root / "data" / "benchmarks" / "synthetic"
LABEL_SUFFIXES = ["_jev_labels.jsonl", "_gpt_5_6_luna_labels.jsonl"]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(items: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _discover_domains() -> list[str]:
    return sorted(
        f.name.replace("_families.jsonl", "")
        for f in DATA_DIR.glob("*_families.jsonl")
    )


def valid_ids_for_domain(domain: str) -> set[str]:
    """Recompute the set of valid item IDs for a domain."""
    fam_path = DATA_DIR / f"{domain}_families.jsonl"
    var_path = DATA_DIR / f"{domain}_variants.jsonl"
    if not fam_path.exists():
        return set()

    families = _load_jsonl(fam_path)
    variants = _load_jsonl(var_path) if var_path.exists() else []

    ids: set[str] = set()
    for idx, fam in enumerate(families):
        family_idx = fam.get("_family_idx", idx)
        vd = variants[idx] if idx < len(variants) else {}
        for it in family_to_typed_questions(fam, vd, domain, family_idx):
            ids.add(it["id"])
    return ids


def clean_label_file(path: Path, valid_ids: set[str], dry_run: bool) -> tuple[int, int]:
    """Remove orphan entries from a label file. Returns (kept, removed)."""
    labels = _load_jsonl(path)
    kept = [lbl for lbl in labels if lbl.get("id") in valid_ids]
    removed = len(labels) - len(kept)

    if removed and not dry_run:
        _write_jsonl(kept, path)

    return len(kept), removed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Remove orphan Jev/Luna label entries not producible by the current pipeline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without modifying files."
    )
    parser.add_argument("--domain", type=str, help="Clean a single domain.")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else _discover_domains()

    logger.info("%-33s %10s %10s %10s", "Domain / File", "Kept", "Removed", "Total")
    logger.info("-" * 70)

    total_removed = 0
    total_kept = 0

    for domain in domains:
        valid_ids = valid_ids_for_domain(domain)

        for suffix in LABEL_SUFFIXES:
            label_path = DATA_DIR / f"{domain}{suffix}"
            if not label_path.exists():
                continue

            kept, removed = clean_label_file(label_path, valid_ids, args.dry_run)
            total_removed += removed
            total_kept += kept

            if removed:
                logger.info(
                    "%-33s %10d %10d %10d",
                    label_path.name,
                    kept,
                    removed,
                    kept + removed,
                )

    logger.info("-" * 70)
    logger.info("TOTAL kept=%d removed=%d", total_kept, total_removed)

    if args.dry_run:
        logger.info("Dry-run mode: no files were modified.")


if __name__ == "__main__":
    main()
