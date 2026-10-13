"""Materialize benchmark jsonl files from HuggingFace parquet caches.

The training config path (`data/sources.py:load_source_items`) reads jsonl from
`data/benchmarks/<name>.jsonl`, while the pipeline loaders in `data/pipeline.py`
read straight from HF parquet. Datasets that were only ever loaded through the
pipeline have no jsonl on disk, so a config that references them by path
silently skips them.

Usage:
    python data/scripts/materialize_benchmarks.py mmlu winogrande piqa ...
    python data/scripts/materialize_benchmarks.py            # all loaders
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from data.pipeline import DATA_DIR, LOADERS


def main() -> int:
    names = sys.argv[1:] or sorted(LOADERS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        loader = LOADERS.get(name)
        if loader is None:
            print(f"  {name}: unknown loader, skipping", file=sys.stderr)
            continue

        path = DATA_DIR / f"{name}.jsonl"
        try:
            items = list(loader())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

        print(f"  {name}: {len(items):,} items -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
