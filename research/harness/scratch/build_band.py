"""Keep the learnability band: drop items the frozen base solves or misses confidently."""
import argparse
import json
from pathlib import Path

from data.pipeline import load_jsonl, save_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    items = {i.id: i for i in load_jsonl(args.items)}
    keep = []
    dropped = {"conf_correct": 0, "conf_wrong": 0}
    for line in args.probs.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        item = items.get(rec["id"])
        if item is None:
            continue
        keys, vals = rec["keys"], rec["probs"]
        top = max(range(len(vals)), key=lambda j: vals[j])
        conf = vals[top]
        if rec["q_type"] == "noul":
            correct = (top == 1) == bool(item.label)
        elif rec["q_type"] == "choice":
            correct = str(keys[top]) == str(item.label)
        else:
            correct = int(item.label) == top
        if conf >= args.threshold:
            dropped["conf_correct" if correct else "conf_wrong"] += 1
            continue
        keep.append(item)
    save_jsonl(keep, args.out)
    print(f"kept {len(keep)} dropped {dropped} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
