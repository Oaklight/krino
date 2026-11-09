"""Second pass of the cascade: judgement scores for the top-2 candidates (#342).

Reads a panel-2 result (per-item code logits) and the matching rows file, takes
the top two candidates by softmax probability, and scores each candidate with
the per-option judgement readout on main's rendering path
(:func:`model.evaluation.judgement_prompt.render`).  Two forwards per eligible
item, one per candidate.

Items the judgement readout cannot name two candidates for — a noul item
without criteria, where the readout has only a polarity branch — are recorded
as ``eligible: false`` and the cascade leaves them on the first-pass answer.

The mapping from a panel-2 natural index to a judgement candidate index is
explicit and checked: for choice and score both readouts use the criteria order,
and for noul panel 2 shows ``no``/``yes`` while the judgement shows the item's
true wording first, so the index inverts.

Usage::

    python model/scripts/score_cascade_pairs.py --rows rows_dev.jsonl \\
        --panel2 panel2_dev.json --dataset dev --dev-path <dev.jsonl> \\
        --snapshot <snapshot> --output pairs_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.cascade import top_two  # noqa: E402
from model.evaluation.dev_set import load_dev_set  # noqa: E402
from model.evaluation.judgement_prompt import (  # noqa: E402
    JUDGEMENT_TEMPLATE,
    NO,
    YES,
    candidates,
    render,
)
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_id(row_id: str) -> str:
    return row_id.split(":", 1)[1] if ":" in row_id else row_id


def judgement_index(q_type: str, natural_index: int, n_options: int) -> int:
    """Panel-2 natural index to judgement candidate index for one item type."""
    if q_type == "noul":
        return n_options - 1 - natural_index
    return natural_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--panel2", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dev", "jevbench"), required=True)
    parser.add_argument("--dev-path", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    panel2 = {row["id"]: row for row in load_json(args.panel2)["predictions_natural"]}

    if args.dataset == "dev":
        if args.dev_path is None:
            raise SystemExit("--dev-path is required for --dataset dev")
        items = list(load_dev_set(args.dev_path))
    else:
        items = list(load_jevbench())
    by_id = {item.id: item for item in items}

    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=torch.bfloat16)
    scorer = LogitScorer(
        model, tokenizer, norm="mean", strategy="description", batch_size=1
    )
    instructions_cache: dict[str, str] = {}

    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        item = by_id[raw_id(row["id"])]
        panel = panel2[row["id"]]
        q_type = row["type"]
        first, second, margin = top_two(panel["code_logits"])
        if first != panel["pred_pos"]:
            raise SystemExit(
                f"{row['id']}: top-1 {first} disagrees with panel-2 pred {panel['pred_pos']}"
            )
        instructions = instructions_cache.setdefault(
            item.id, item.question.get("instructions", "")
        )
        prepared = candidates(item)
        record = {
            "id": row["id"],
            "type": q_type,
            "gold": int(row["gold"]),
            "base": int(panel["pred_pos"]),
            "candidates": [first, second],
            "margin": margin,
            "eligible": prepared is not None,
            "scores": {},
            "latency_s": 0.0,
        }
        if prepared is not None:
            texts, gold_index = prepared
            if q_type != "noul":
                mapped = judgement_index(q_type, first, len(texts))
                if texts[mapped] != row["texts"][first]:
                    raise SystemExit(
                        f"{row['id']}: judgement text {texts[mapped]!r} does not match "
                        f"panel-2 text {row['texts'][first]!r}"
                    )
            for natural in (first, second):
                candidate_index = judgement_index(q_type, natural, len(texts))
                started_forward = time.perf_counter()
                with torch.inference_mode():
                    scores = scorer.score_options(
                        render(item.state, instructions, texts[candidate_index]),
                        [YES, NO],
                    )
                record["latency_s"] += time.perf_counter() - started_forward
                record["scores"][natural] = float(scores[0])
        results.append(record)
        if (index + 1) % 50 == 0:
            print(
                f"  {index + 1}/{len(rows)} items, {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    eligible = sum(1 for row in results if row["eligible"])
    payload = {
        "config": {
            "dataset": args.dataset,
            "judgement_template": JUDGEMENT_TEMPLATE,
            "judgement_norm": "mean",
            "judgement_strategy": "description",
            "scores_compared": "log P(' yes') per candidate",
            "tie_break": "smaller natural option index",
            "margin": "softmax over the K panel-2 code logits, p1 - p2",
            "noul_index_map": "inverted (panel 2 no/yes, judgement true/false)",
        },
        "items": len(results),
        "eligible": eligible,
        "ineligible": len(results) - eligible,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"written {args.output}: {len(results)} items, {eligible} eligible, "
        f"{payload['elapsed_s']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
