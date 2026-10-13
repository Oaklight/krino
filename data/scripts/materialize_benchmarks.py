"""Materialize benchmark jsonl files from HuggingFace parquet caches.

The training config path (`data/sources.py:load_source_items`) reads jsonl from
`data/benchmarks/<name>.jsonl`, while the pipeline loaders in `data/pipeline.py`
read straight from HF parquet. Datasets that were only ever loaded through the
pipeline have no jsonl on disk, so a config that references them by path
silently skips them.

Usage:
    python data/scripts/materialize_benchmarks.py mmlu winogrande piqa ...
    python data/scripts/materialize_benchmarks.py            # all loaders

Exits non-zero if any requested loader failed or produced no items, so that a
caller cannot mistake a partial run for a complete one.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from data.pipeline import DATA_DIR, LOADERS


def _summarise(items: list) -> str:
    """Per-split counts, so an eval-only source is visible at a glance."""
    counts = Counter(item.split for item in items)
    return " ".join(f"{name}={count:,}" for name, count in sorted(counts.items()))


def main() -> int:
    names = sys.argv[1:] or sorted(LOADERS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []

    for name in names:
        loader = LOADERS.get(name)
        if loader is None:
            print(f"  {name}: unknown loader", file=sys.stderr)
            failed.append(name)
            continue

        path = DATA_DIR / f"{name}.jsonl"

        try:
            items = list(loader())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            failed.append(name)
            continue

        if not items:
            # An empty file is worse than no file: load_source_items only warns
            # when the path is missing, so an empty jsonl reads as a source
            # contributing zero items with no signal at all.
            print(
                f"  {name}: loader yielded 0 items, not writing {path.name}",
                file=sys.stderr,
            )
            failed.append(name)
            continue

        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

        print(f"  {name}: {len(items):,} items ({_summarise(items)}) -> {path}")

    if failed:
        print(
            f"\n{len(failed)} source(s) not written: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
