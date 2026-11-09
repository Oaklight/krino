"""Emit panel-2 rows for the set-head arm (#340).

One JSONL line per item, holding everything the extractor needs and nothing it
must recompute:

``{"id", "source", "split", "type", "state", "instructions", "texts", "gold"}``

The corpus arm is the #324 canonical source list with its caps and seed 42, the
same construction that produced the judgement plan, so the option-state cache
is over the same item set.  An item is kept only when panel 2 can read it:
:func:`options_for` returns options and a gold index, and the option count is
inside the 26-code range.  Out-of-range and unscorable items are counted, not
silently dropped.

Usage::

    python model/scripts/build_set_head_rows.py --dataset corpus --out rows.jsonl
    python model/scripts/build_set_head_rows.py --dataset dev --dev-path <dev.jsonl> --out rows.jsonl
    python model/scripts/build_set_head_rows.py --dataset jevbench --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _vendor import yaml  # noqa: E402

from data.pipeline import load_jevbench  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.evaluation.dev_set import load_dev_set  # noqa: E402
from model.evaluation.panel2 import CODES, options_for  # noqa: E402
from model.evaluation.panel2_states import data_digest  # noqa: E402

SEED = 42
CONFIG = "model/configs/r2_reranker4b.yaml"


def row(item, source: str, split: str) -> dict | None:
    """The cache row for ``item``, or ``None`` when panel 2 cannot read it."""
    prepared = options_for(item)
    if prepared is None:
        return None
    texts, gold = prepared
    if not 2 <= len(texts) <= len(CODES):
        return None
    return {
        "id": f"{source}:{item.id}",
        "source": source,
        "split": split,
        "type": item.question.get("type", "noul"),
        "state": item.state,
        "instructions": item.question.get("instructions", ""),
        "texts": list(texts),
        "gold": int(gold),
    }


def corpus_rows(limit: int | None) -> list[dict]:
    config = yaml.load(open(CONFIG, encoding="utf-8").read())
    sources = config["sources"]
    rng = random.Random(SEED)
    rows: list[dict] = []
    counts: Counter = Counter()
    for name, raw_cfg in sources.items():
        cfg = dict(raw_cfg)
        train_items, eval_items = load_source_items(name, cfg, rng)
        for split, items in (("train", train_items), ("test", eval_items)):
            for item in items:
                if getattr(item, "answerable", True) is False:
                    counts["skipped:abstention"] += 1
                    continue
                built = row(item, name, split)
                if built is None:
                    counts[f"skipped:{split}"] += 1
                    continue
                counts[f"{split}:items"] += 1
                counts[f"{split}:options"] += len(built["texts"])
                rows.append(built)
        if limit is not None and len(rows) >= limit:
            break
    print(dict(sorted(counts.items())), flush=True)
    return rows[:limit] if limit else rows


def named_rows(dataset: str, dev_path: str | None, limit: int | None) -> list[dict]:
    if dataset == "jevbench":
        items = list(load_jevbench())
        rows = [built for item in items if (built := row(item, "jevbench", "jevbench"))]
    else:
        if not dev_path:
            raise SystemExit("--dev-path is required for --dataset dev")
        items = list(load_dev_set(dev_path))
        rows = [built for item in items if (built := row(item, "dev", "dev"))]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("corpus", "dev", "jevbench"), required=True
    )
    parser.add_argument("--dev-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "corpus":
        rows = corpus_rows(args.limit)
    else:
        rows = named_rows(args.dataset, args.dev_path, args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for built in rows:
            handle.write(json.dumps(built, ensure_ascii=False) + "\n")

    splits = Counter(built["split"] for built in rows)
    types = Counter(built["type"] for built in rows)
    print(
        f"wrote {len(rows)} rows to {args.out} splits={dict(splits)} "
        f"types={dict(types)}",
        flush=True,
    )
    print(f"data_digest {data_digest(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
