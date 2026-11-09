"""Score panel 2 over an arbitrary rows file (#340).

The set-head comparison needs panel 2 on the development set, and main's
harness only scores JevBench.  This driver reads the same rows file the state
extractor used, calls the shipped ``score_prompt`` / ``choose_option`` path
unchanged, and writes the two prediction arrays in the shape
``eval_set_head.py --panel2`` reads.

On JevBench the output must reproduce the committed panel-2 file; that is the
check that this driver measures panel 2 and not a reimplementation.

Usage::

    python model/scripts/score_panel2_rows.py --snapshot <snapshot> \\
        --rows rows_dev.jsonl --output results/jevbench/panel2_dev_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_jevbench_panel2 import score_prompt  # noqa: E402
from model.evaluation.panel2 import (  # noqa: E402
    code_token_ids,
    codes_for,
    has_exact_tie,
    option_index_for_display,
    render_panel2,
)
from model.src.backbone import load_causal_lm  # noqa: E402

CODE_FORM = "space"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_order(
    model, tokenizer, rows: list[dict], order: str, device: str
) -> list[dict]:
    """Panel-2 predictions with ``pred_pos`` always a natural option index."""
    predictions = []
    code_ids_by_k: dict[int, dict[str, int]] = {}
    for row in rows:
        texts = list(row["texts"])
        gold = int(row["gold"])
        reversed_order = order == "reversed"
        if reversed_order:
            texts = list(reversed(texts))
        codes = codes_for(len(texts))
        if len(texts) not in code_ids_by_k:
            code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
            if multi:
                raise SystemExit(f"codes {multi} are not single tokens")
            code_ids_by_k[len(texts)] = code_ids
        code_ids = code_ids_by_k[len(texts)]
        prompt = render_panel2(row["state"], row["instructions"], texts, codes)
        display_to_option = option_index_for_display(len(texts), reversed_order)
        values, chosen, elapsed, _ = score_prompt(
            model, tokenizer, prompt, codes, code_ids, device, display_to_option
        )
        # Same coordinates as the shipped harness: ``pred_pos`` is a display
        # position and a reversed row's ``gold_pos`` is display too, so the two
        # files can be compared row by row.
        prediction = display_to_option.index(chosen)
        gold_pos = len(texts) - 1 - gold if reversed_order else gold
        predictions.append(
            {
                "id": row["id"],
                "type": row["type"],
                "order": order,
                "n_options": len(texts),
                "gold_pos": gold_pos,
                "pred_pos": prediction,
                "correct": prediction == gold_pos,
                "code_logits": values,
                "tie": bool(has_exact_tie(values)),
                "latency_s": elapsed,
            }
        )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=torch.bfloat16)
    device = str(next(model.parameters()).device)
    result = {}
    for order in ("natural", "reversed"):
        predictions = score_order(model, tokenizer, rows, order, device)
        correct = sum(row["correct"] for row in predictions)
        result[f"predictions_{order}"] = predictions
        result[f"metrics_{order}"] = {
            "correct": correct,
            "total": len(predictions),
            "accuracy": correct / len(predictions),
        }
        print(f"{order}: {correct}/{len(predictions)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
