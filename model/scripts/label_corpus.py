"""Attach Jev teacher probabilities to a TypedQuestion JSONL, resumably.

Writes a checkpoint every ``--checkpoint-every`` items (atomic: temp file +
rename), so an interrupted run loses at most one chunk.  Resume keys on
``teacher_probs["jev"]`` already present in ``--out``.

Augmented noul items are skipped: ``label_items_sync`` sends them to Jev as a
choice over the augmented label set, but ``train_residual``'s noul soft target
expects ``false``/``true`` and would normalise the wrong subset.  Those items
keep their hard labels.

The sidecar ``--stats`` JSON reports the teacher/hard-label agreement rate
(overall and by type), the score off-by-one rate, how many teacher
distributions line up with the option keys ``train_residual`` consumes, and
the confidence-bucket mix.

Usage:
    python model/scripts/label_corpus.py \
        --items data/benchmarks/residual_train_20000.jsonl \
        --out data/benchmarks/residual_train_20000_jev.jsonl \
        --batch-size 10 --checkpoint-every 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for search_path in (str(ROOT / "probing" / "scripts"), str(ROOT)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from jev_client import JevClient  # noqa: E402

from data.pipeline import load_jsonl, save_jsonl  # noqa: E402
from data.synthetic_label import (  # noqa: E402
    CONFIDENCE_BUCKETS,
    label_items_sync,
    print_bucket_report,
)


def _option_keys(item) -> list[str]:
    q = item.question
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return [str(key) for key in q["criteria"]]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return []


def _hard_key(item) -> str | None:
    q = item.question
    if q["type"] == "noul":
        return "true" if bool(item.label) else "false"
    if q["type"] == "choice":
        return str(item.label)
    if q["type"] == "score":
        try:
            return str(int(round(float(item.label))))
        except (TypeError, ValueError):
            return None
    return None


def _needs_label(item) -> bool:
    """Augmented noul is sent as choice; train_residual cannot consume it."""
    q = item.question
    return not (q["type"] == "noul" and q.get("augmented_options"))


def _proxy_bucket(confidence: float) -> str:
    for bucket, (low, high) in CONFIDENCE_BUCKETS.items():
        if low <= confidence < high:
            return bucket
    return "uncertain"


def _summary(counts: Counter, q_type: str = "") -> dict:
    decided = counts["agree"] + counts["disagree"]
    out = {
        "labeled": counts["labeled"],
        "usable": counts["usable"],
        "unusable": counts["unusable"],
        "no_hard": counts["no_hard"],
        "agree": counts["agree"],
        "disagree": counts["disagree"],
        "agreement_rate": counts["agree"] / decided if decided else None,
        "usable_rate": (
            counts["usable"] / counts["labeled"] if counts["labeled"] else None
        ),
    }
    if q_type == "score":
        out["within_one"] = counts["within_one"]
        out["within_one_rate"] = (
            counts["within_one"] / decided if decided else None
        )
    return out


def _agreement(items, teacher: str) -> dict:
    """Teacher argmax vs hard label over the option keys the arm scores."""
    overall: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        probs = (item.teacher_probs or {}).get(teacher) or {}
        if not probs or not _needs_label(item):
            continue
        q_type = item.question["type"]
        overall["labeled"] += 1
        by_type[q_type]["labeled"] += 1

        keys = _option_keys(item)
        if not keys or any(key not in probs for key in keys):
            overall["unusable"] += 1
            by_type[q_type]["unusable"] += 1
            continue
        overall["usable"] += 1
        by_type[q_type]["usable"] += 1

        hard = _hard_key(item)
        if hard is None or hard not in keys:
            overall["no_hard"] += 1
            by_type[q_type]["no_hard"] += 1
            continue

        predicted = max(keys, key=lambda key: float(probs[key]))
        if predicted == hard:
            overall["agree"] += 1
            by_type[q_type]["agree"] += 1
        else:
            overall["disagree"] += 1
            by_type[q_type]["disagree"] += 1

        if q_type == "score":
            try:
                if abs(int(predicted) - int(hard)) <= 1:
                    overall["within_one"] += 1
                    by_type[q_type]["within_one"] += 1
            except (TypeError, ValueError):
                pass

    return {
        "overall": _summary(overall),
        "by_type": {
            q_type: _summary(counts, q_type)
            for q_type, counts in sorted(by_type.items())
        },
    }


def _proxy_buckets(items, teacher: str) -> dict:
    counts: Counter = Counter()
    for item in items:
        probs = (item.teacher_probs or {}).get(teacher) or {}
        if not probs or not _needs_label(item):
            continue
        counts[_proxy_bucket(max(float(value) for value in probs.values()))] += 1
    return dict(counts)


def _distribution_sums(items, teacher: str) -> dict | None:
    sums = []
    for item in items:
        probs = (item.teacher_probs or {}).get(teacher) or {}
        keys = _option_keys(item)
        if not probs or not keys or any(key not in probs for key in keys):
            continue
        sums.append(sum(float(probs[key]) for key in keys))
    if not sums:
        return None
    ordered = sorted(sums)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p1": ordered[max(0, len(ordered) // 100)],
        "p50": ordered[len(ordered) // 2],
        "p99": ordered[min(len(ordered) - 1, len(ordered) * 99 // 100)],
        "max": ordered[-1],
        "below_0.9": sum(value < 0.9 for value in ordered),
    }


def _out_items(
    items,
    existing: dict[str, object],
    done: set[str],
    labeled: dict[str, dict],
    teacher: str,
):
    out = []
    for item in items:
        base = existing.get(item.id, item)
        if item.id in done:
            out.append(base)
            continue
        probs = labeled.get(item.id)
        if probs and _needs_label(base):
            merged = dict(base.teacher_probs or {})
            merged[teacher] = probs
            base.teacher_probs = merged
            base.teacher_name = teacher
        out.append(base)
    return out


def _write_atomic(items, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    save_jsonl(items, temporary)
    temporary.replace(path)


def _chunks_by_state(items, max_items: int):
    """Keep each state in one chunk so label_items_sync does not re-ask it."""
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(item.state, []).append(item)
    chunk: list = []
    for state_items in groups.values():
        if chunk and len(chunk) + len(state_items) > max_items:
            yield chunk
            chunk = []
        chunk.extend(state_items)
    if chunk:
        yield chunk


def _state_groups(items) -> list[list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(item.state, []).append(item)
    return list(groups.values())


def _payload(items) -> list[dict]:
    return [
        {"id": item.id, "state": item.state, "question": item.question}
        for item in items
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--teacher", default="jev")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Write the output file after roughly this many items",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="States to label in parallel; 1 keeps the sequential path",
    )
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")

    items = load_jsonl(args.items)
    existing: dict[str, object] = {}
    done: set[str] = set()
    if args.out.exists():
        for item in load_jsonl(args.out):
            existing[item.id] = item
            if item.teacher_probs and args.teacher in item.teacher_probs:
                done.add(item.id)
    skipped_augmented = sum(1 for item in items if not _needs_label(item))
    todo = [
        item
        for item in items
        if item.id not in done and _needs_label(item)
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"items {len(items)} done {len(done)} todo {len(todo)} "
        f"skipped_augmented {skipped_augmented}",
        flush=True,
    )

    results = []
    labeled: dict[str, dict] = {}
    client = JevClient() if todo else None
    try:
        for chunk in _chunks_by_state(todo, args.checkpoint_every):
            if args.concurrency == 1:
                chunk_results = label_items_sync(
                    _payload(chunk), client, batch_size=args.batch_size
                )
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    chunk_results = [
                        result
                        for group_results in pool.map(
                            lambda group: label_items_sync(
                                _payload(group), client, batch_size=1
                            ),
                            _state_groups(chunk),
                        )
                        for result in group_results
                    ]
            results.extend(chunk_results)
            for result in chunk_results:
                if result.teacher_probs:
                    labeled[result.question_id] = result.teacher_probs
            _write_atomic(
                _out_items(items, existing, done, labeled, args.teacher), args.out
            )
            print(
                f"checkpoint {len(labeled)}/{len(todo)} labeled -> {args.out}",
                flush=True,
            )
    finally:
        if client is not None:
            client.close()

    out = _out_items(items, existing, done, labeled, args.teacher)
    _write_atomic(out, args.out)
    if results:
        print_bucket_report(results)

    stats = {
        "items": str(args.items),
        "out": str(args.out),
        "teacher": args.teacher,
        "input": len(items),
        "already_done": len(done),
        "todo": len(todo),
        "skipped_augmented": skipped_augmented,
        "concurrency": args.concurrency,
        "labeled_this_run": len(labeled),
        "failed_this_run": max(0, len(todo) - len(labeled)),
        "labeled_total": sum(
            1
            for item in out
            if (item.teacher_probs or {}).get(args.teacher) and _needs_label(item)
        ),
        "agreement": _agreement(out, args.teacher),
        "distribution_sums": _distribution_sums(out, args.teacher),
        "confidence_buckets_this_run": dict(
            Counter(result.bucket for result in results)
        ),
        "confidence_buckets_proxy_all": _proxy_buckets(out, args.teacher),
    }
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"labeled {len(labeled)}/{len(todo)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
