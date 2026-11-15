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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _vendor import yaml  # noqa: E402

from data.pipeline import load_jsonl  # noqa: E402
from data.sources import load_source_items  # noqa: E402
from model.src.judgement_objective import build_prompts  # noqa: E402
from model.training.group_split import (  # noqa: E402
    load_cluster_manifest,
    split_items_by_group,
    state_key,
)

SYNTHETIC_TEMPLATE = re.compile(r"^synthetic-(.+)-(\d{4})-")


def load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text())


def load_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())
    values = payload["ids"] if isinstance(payload, dict) else payload
    return {str(value) for value in values}


def source_of(item, fallback: str) -> str:
    """The item's own source; ``corpus`` is a mode name, not a source."""
    value = getattr(item, "source", None)
    return value if isinstance(value, str) and value else fallback


def synthetic_family(item) -> str | None:
    """The family embedded in a synthetic group key (``synthetic-<family>-NNNN``)."""
    if source_of(item, "") != "synthetic":
        return None
    group = getattr(item, "group", "") or ""
    parts = group.split("-")
    if len(parts) > 1 and parts[0] == "synthetic":
        return parts[1]
    return None


def synthetic_template(item) -> tuple[str, str] | None:
    """The generated item behind a synthetic row: ``(family, 4-digit item id)``."""
    if source_of(item, "") != "synthetic":
        return None
    match = SYNTHETIC_TEMPLATE.match(str(getattr(item, "id", "")))
    return (match.group(1), match.group(2)) if match else None


def namespace_of(item) -> str:
    """Split namespace: source, or source+family for synthetic rows."""
    source = source_of(item, "")
    family = synthetic_family(item)
    return f"synthetic/{family}" if source == "synthetic" and family else source


def plan_items(args, entries):
    """Yield ``(name, item, row_id, split, answerable, prompts)`` for one plan row."""
    for name, items in entries:
        for item in items:
            split = getattr(item, "split", "train")
            row_id = f"{name}:{item.id}"
            answerable = getattr(item, "answerable", True) is not False
            built = build_prompts(
                item,
                row_id=row_id,
                source=source_of(item, name),
                split=split,
                answerable=answerable,
            )
            yield name, item, row_id, split, answerable, built


def internal_labels(args, entries) -> tuple[dict[str, str], dict] | tuple[None, None]:
    """Compute the corrected split and the per-row ``internal_split`` labels."""
    if not args.internal_split:
        return None, None
    clusters = load_cluster_manifest(args.clusters) if args.clusters else None
    records: list[tuple[str, object, int, str, str]] = []
    for _name, item, row_id, split, _answerable, built in plan_items(args, entries):
        if not built:
            continue
        q_type = item.question.get("type", "noul")
        records.append((row_id, item, len(built), q_type, split))

    def cluster_of(record):
        if clusters is None:
            return None
        item = record[1]
        return clusters.get(f"id:{item.id}") or clusters.get(
            f"state:{state_key(item.state)}"
        )

    pool = [record for record in records if record[4] == "train"]
    train, val, report = split_items_by_group(
        pool,
        args.val_fraction,
        args.split_seed,
        id_of=lambda record: record[0],
        state_of=lambda record: state_key(record[1].state),
        group_of=lambda record: getattr(record[1], "group", None),
        cluster_of=cluster_of,
        stratum_of=lambda record: (record[3], min(record[2], 8)),
        namespace_of=lambda record: namespace_of(record[1]),
        use_group=False,
        force_train_clusters=args.force_train_cluster,
    )
    if args.cap_per_template:
        train = cap_per_template(train, args.cap_per_template)
    if args.subset_rows:
        order = list(range(len(train)))
        random.Random(args.subset_seed).shuffle(order)
        keep = sorted(order[: args.subset_rows])
        train = [train[index] for index in keep]
        labels = {record[0]: "train" for record in train}
    else:
        labels = {record[0]: "train" for record in train}
        labels.update({record[0]: "val" for record in val})
        for record in records:
            if record[4] != "train":
                labels.setdefault(record[0], "external")
    report = dict(report)
    report["train_after_cap"] = len(train)
    report["val_after_cap"] = len(val)
    return labels, report


def cap_per_template(records, cap: int):
    """Keep at most ``cap`` rows per synthetic template, round-robin over q_type.

    The kept rows are the lexicographically smallest row ids per q_type, so the
    result is deterministic and keeps every question type the template has.
    """
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    keep = []
    for record in records:
        template = synthetic_template(record[1])
        if template is None:
            keep.append(record)
        else:
            groups[template][record[3]].append(record)
    for by_type in groups.values():
        for q_type in by_type:
            by_type[q_type].sort(key=lambda record: record[0])
        picked = []
        index = 0
        while len(picked) < cap:
            added = False
            for q_type in sorted(by_type):
                if index < len(by_type[q_type]):
                    picked.append(by_type[q_type][index])
                    added = True
                    if len(picked) == cap:
                        break
            if not added:
                break
            index += 1
        keep.extend(picked)
    kept = {record[0] for record in keep}
    return [record for record in records if record[0] in kept]


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
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="near-duplicate cluster manifest for --internal-split",
    )
    parser.add_argument(
        "--internal-split",
        action="store_true",
        help="compute the corrected train/val split and write internal_split on every row",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--force-train-cluster",
        action="append",
        default=[],
        help="cluster id whose component must stay in train (repeatable)",
    )
    parser.add_argument(
        "--cap-per-template",
        type=int,
        default=None,
        help="keep at most this many train rows per synthetic template",
    )
    parser.add_argument(
        "--subset-rows",
        type=int,
        default=None,
        help="draw this many rows once from the capped train set and keep only those",
    )
    parser.add_argument("--subset-seed", type=int, default=42)
    args = parser.parse_args()
    if args.cap_per_template and not args.internal_split:
        parser.error("--cap-per-template requires --internal-split")
    if args.subset_rows and not args.internal_split:
        parser.error("--subset-rows requires --internal-split")

    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    k_hist: Counter[int] = Counter()
    lengths: list[int] = []
    rows = 0
    prompts = 0
    skipped_items = 0
    masked_rows = 0

    entries = corpus_items(args, rng)
    labels, split_report = internal_labels(args, entries)
    if split_report is not None:
        print(f"internal split {json.dumps(split_report, sort_keys=True)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        current = None
        kept = 0
        for name, item, row_id, split, answerable, built in plan_items(args, entries):
            if name != current:
                if current is not None:
                    print(f"{current:22} rows {kept:7,}", flush=True)
                current = name
                kept = 0
            if labels is not None and row_id not in labels:
                continue
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
                record["family"] = synthetic_family(item)
                if labels is not None:
                    record["internal_split"] = labels[row_id]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                prompts += 1
                lengths.append(len(prompt.text))
        if current is not None:
            print(f"{current:22} rows {kept:7,}", flush=True)

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
