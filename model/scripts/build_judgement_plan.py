#!/usr/bin/env python3
"""Build the judgement-objective prompt plan from the canonical corpus (#324).

One JSONL line per option prompt, produced by
``model/src/judgement_objective.py::build_prompts``, which itself calls the
same ``candidates``/``render`` the readout scores.  Training and inference
therefore share one definition (``#299``).

Two input modes:

* ``--corpus PATH``: a canonical ``TypedQuestion`` JSONL (the frozen artifact).
  This is the training path.  ``--holdout-ids`` removes the held-out ids, which
  are never re-split.
* ``--config PATH``: the source list of ``model/configs/r2_reranker4b.yaml``
  with the per-source ``max_train`` caps removed unless ``--capped`` is passed;
  that is the 243,366-row corpus ``#299`` measured.

Abstention rows (``answerable = False``) are kept and marked, never dropped: the
trainer masks their loss (#254), so the plan's mixture and denominator match the
corpus while the rows add no gradient.  Eval-only sources (``weight: 0``) are
kept out of the train split.

Usage:
    python model/scripts/build_judgement_plan.py \\
        --corpus data/benchmarks/krino-canonical-234285.jsonl \\
        --holdout-ids heldout_6400.json \\
        --out data/benchmarks/judgement_plan.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _vendor import yaml  # noqa: E402

from data.pipeline import load_jsonl  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.src.judgement_objective import build_prompts  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text())


def load_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())
    values = payload["ids"] if isinstance(payload, dict) else payload
    return {str(value) for value in values}


def corpus_items(args, rng: random.Random) -> list[tuple[str, list]]:
    """The two input modes as ``(source name, items)`` pairs."""
    if args.corpus:
        items = load_jsonl(args.corpus)
        holdout = load_ids(args.holdout_ids)
        if holdout:
            items = [item for item in items if str(item.id) not in holdout]
            print(f"excluded {len(holdout)} held-out ids", flush=True)
        if args.sample:
            rng.shuffle(items)
            items = items[: args.sample]
        return [("corpus", items)]

    config = load_yaml(args.config)
    sources: dict[str, dict] = config.get("sources", {})
    if not sources:
        raise SystemExit(f"{args.config} has no sources")
    loaded = []
    for name, raw_cfg in sources.items():
        cfg = dict(raw_cfg)
        if not args.capped:
            cfg.pop("max_train", None)
            cfg.pop("max_eval", None)
        train_items, eval_items = load_source_items(name, cfg, rng)
        items = train_items + eval_items
        if args.max_rows_per_source is not None:
            items = items[: args.max_rows_per_source]
        loaded.append((name, items))
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="canonical TypedQuestion JSONL; overrides --config",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "model/configs/r2_reranker4b.yaml"
    )
    parser.add_argument(
        "--holdout-ids",
        type=Path,
        default=None,
        help="JSON id list (or {'ids': [...]}) excluded from the plan",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="corpus mode: shuffle and keep this many rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capped",
        action="store_true",
        help="keep each source's max_train/max_eval caps instead of the full split",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="debug aid: keep at most this many rows per source",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    k_hist: Counter[int] = Counter()
    lengths: list[int] = []
    rows = 0
    prompts = 0
    skipped_items = 0
    masked_rows = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for name, items in corpus_items(args, rng):
            kept = 0
            for item in items:
                split = getattr(item, "split", "train")
                row_id = f"{name}:{item.id}"
                answerable = getattr(item, "answerable", True) is not False
                built = build_prompts(
                    item,
                    row_id=row_id,
                    source=name,
                    split=split,
                    answerable=answerable,
                )
                if not built:
                    skipped_items += 1
                    continue
                kept += 1
                rows += 1
                k_hist[len(built)] += 1
                if not answerable:
                    masked_rows += 1
                    counts[f"{split}:masked"] += 1
                counts[f"{split}:{item.question.get('type', 'noul')}"] += 1
                if len(built) == 1 and not built[0].has_option_span:
                    counts[f"{split}:polarity"] += 1
                for prompt in built:
                    record = dataclasses.asdict(prompt)
                    record["label"] = prompt.label
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    prompts += 1
                    lengths.append(len(prompt.text))
            print(f"{name:22} rows {kept:7,}", flush=True)

    print(f"wrote {args.out}")
    print(
        f"rows {rows:,}  prompts {prompts:,}  items skipped {skipped_items:,}  "
        f"masked rows {masked_rows:,}"
    )
    if rows:
        print(f"mean options per row {prompts / rows:.2f}")
    if lengths:
        ordered = sorted(lengths)
        quantiles = {
            q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
            for q in (0.5, 0.9, 0.99)
        }
        print(f"prompt characters {quantiles} max {ordered[-1]}")
    print(f"option-count histogram {dict(sorted(k_hist.items()))}")
    for key in sorted(counts):
        print(f"  {key:20} {counts[key]:8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
