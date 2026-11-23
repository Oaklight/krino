"""Attach Jev teacher probabilities to a TypedQuestion JSONL, resumably."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "probing" / "scripts"))

from jev_client import JevClient  # noqa: E402

from data.pipeline import load_jsonl, save_jsonl  # noqa: E402
from data.synthetic_label import label_items_sync, print_bucket_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--teacher", default="jev")
    args = parser.parse_args()

    items = load_jsonl(args.items)
    done = set()
    if args.out.exists():
        for item in load_jsonl(args.out):
            if item.teacher_probs and args.teacher in item.teacher_probs:
                done.add(item.id)
    todo = [item for item in items if item.id not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"items {len(items)} done {len(done)} todo {len(todo)}", flush=True)

    results = []
    if todo:
        payload = [
            {"id": item.id, "state": item.state, "question": item.question}
            for item in todo
        ]
        client = JevClient()
        try:
            results = label_items_sync(payload, client, batch_size=args.batch_size)
        finally:
            client.close()
        print_bucket_report(results)

    by_id = {r.question_id: r for r in results}
    out = []
    labeled = 0
    for item in items:
        if item.id in done:
            out.append(item)
            continue
        result = by_id.get(item.id)
        if result and result.teacher_probs:
            merged = dict(item.teacher_probs or {})
            merged[args.teacher] = result.teacher_probs
            item.teacher_probs = merged
            item.teacher_name = args.teacher
            labeled += 1
        out.append(item)
    save_jsonl(out, args.out)
    print(f"labeled {labeled}/{len(todo)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
