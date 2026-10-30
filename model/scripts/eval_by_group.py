#!/usr/bin/env python3
"""Per-group accuracy for one checkpoint, on a source that carries group tags.

The eval harness reports by source and by question type. The authored held-out set
is a single source with five families under `group`, so a single accuracy hides
which family a checkpoint fails — and on the first scoring run it hid two families
the checkpoints were not reading at all: one where every checkpoint answered "yes"
four times in five, and one where a single level was predicted for all 51 items.

This scores each item with the same heads the eval harness uses and groups the
result by `TypedQuestion.group`, printing the prediction histogram beside each
accuracy: an accuracy near chance can be a weak model or a constant answer, and
the histogram says which. `--dump-preds` writes the per-item gold/prediction pairs
for the cases the histogram cannot settle — identical histograms across different
checkpoints look like a shared prior, but only the per-item matrix shows whether
the mapping has structure (a rotation, a threshold read one bracket off) or is
diffuse.

Usage:
    python model/scripts/eval_by_group.py \
        --config model/configs/eval_authored.yaml \
        --load-heads model/checkpoints/arch_4b_xenc_split138_s123/epoch_005.pt \
        --source authored_holdout \
        --choice-mode cross_encoder
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

import data.pipeline as pipeline_mod  # noqa: E402
import data.sources as sources_mod  # noqa: E402
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info  # noqa: E402
from model.src.model_factory import build_decision_model  # noqa: E402
from model.training.supervised import _eval_item  # noqa: E402


def _predict(model: Any, item: Any) -> Any:
    """One prediction in the item's own label type."""
    question = item.question
    kind = question.get("type")
    state = item.state
    instructions = question.get("instructions", "")
    if kind == "choice":
        criteria = question["criteria"]
        keys = list(criteria)
        texts = [value or key for key, value in criteria.items()]
        logits = model.forward_choice(state, instructions, texts)[0].float().cpu()
        return keys[int(logits.argmax())]
    if kind == "noul":
        # `_eval_item` coerces a string label with `label.lower() in ("true", ...)`,
        # so a prediction has to be a bool to compare against either form.
        return bool(torch.sigmoid(model.forward_noul(state, instructions)).item() > 0.5)
    logits = model.forward_score(state, instructions, question["criteria"]).float().cpu()
    return int(logits.argmax())


def _gold(item: Any) -> Any:
    """The label in the type the prediction will be in, as `_eval_item` reads it."""
    label = item.label
    if item.question.get("type") == "noul" and isinstance(label, str):
        return label.lower() in ("true", "yes", "1")
    if item.question.get("type") == "score":
        return int(round(float(label)))
    return label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--source", default="authored_holdout")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--readout-n", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--noul-rank", type=int, default=None)
    parser.add_argument(
        "--choice-mode",
        choices=["tower", "cross_encoder"],
        default=None,
        help=(
            "How the choice head scores options. Each shape has its own parameter "
            "names, so `load_heads` refuses a checkpoint trained under the other "
            "one rather than scoring an untrained head."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--blank-state",
        action="store_true",
        help=(
            "Score every item with an empty state, keeping the instructions and the "
            "options. A family whose accuracy does not move was never being read: the "
            "answer came from the option texts and the instructions alone. A family "
            "that collapses was being read. This is the ablation that separates the "
            "two, and it is also a check on the set itself — if blanking does not "
            "hurt, the items are not testing what they claim."
        ),
    )
    parser.add_argument(
        "--dump-preds",
        type=Path,
        default=None,
        help=(
            "Write one JSON object per item (id, group, type, gold, pred, correct). "
            "The per-group accuracy says a family fails; the predictions say how, "
            "which is the difference between a weak model and one reading a "
            "threshold ladder backwards."
        ),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_name = args.model or cfg.get("model")
    max_length = int(args.max_length or cfg.get("max_length") or 32768)
    readout = args.readout or cfg.get("readout", "last")
    rank = int(args.rank or cfg.get("rank") or 128)
    # Mirrors `train_multitask`: absent means a linear noul head, not "same as rank".
    # Getting this wrong is not visible on its own — the checkpoint simply cannot
    # fill the head — which is why `load_heads` now refuses the mismatch.
    noul_rank = args.noul_rank if args.noul_rank is not None else cfg.get("noul_rank")
    choice_mode = args.choice_mode or str(cfg.get("choice_mode") or "tower")
    lora_layers = args.lora_layers if args.lora_layers is not None else cfg.get("lora_layers")

    if args.data_root:
        sources_mod.ROOT = args.data_root
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"

    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=max_length)
    print_gpu_info(gpu_config)

    model = build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=args.device or cfg.get("device"),
        use_encoder=bool(cfg.get("encoder", False)),
        rank=rank,
        mlp_layers=int(cfg.get("mlp_layers", 0)),
        noul_rank=noul_rank,
        max_length=max_length,
        shared_attention=bool(cfg.get("shared_attention", True)),
        readout=readout,
        readout_n=int(args.readout_n or cfg.get("readout_n") or 4),
        lora_layers=lora_layers,
        lora_rank=int(cfg.get("lora_rank") or 16),
        lora_alpha=int(cfg.get("lora_alpha") or 32),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        lora_targets=tuple(
            part.strip()
            for part in str(cfg.get("lora_targets") or "q_proj,v_proj").split(",")
            if part.strip()
        ),
        choice_mode=choice_mode,
    )
    model.load_heads(args.load_heads)
    model.eval()

    source_cfg = (cfg.get("sources") or {}).get(args.source) or {}
    # Mirror the eval harness first: `load_source_items` is what scores the run, so
    # grouping anything else would describe a different set. Fall back to the
    # pipeline registry only for a source with no prepared file.
    _, items = sources_mod.load_source_items(args.source, source_cfg, random.Random(0))
    if not items:
        loader = pipeline_mod.LOADERS.get(args.source)
        if loader is None:
            print(f"no items for source {args.source!r}", file=sys.stderr)
            return 1
        items = list(loader())

    correct: dict[tuple[Any, Any], list[int]] = collections.defaultdict(lambda: [0, 0])
    preds: dict[tuple[Any, Any], collections.Counter] = collections.defaultdict(collections.Counter)
    disagreements = 0
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for item in items:
            if args.blank_state:
                # Blank the item before either scorer sees it, so the harness
                # cross-check below still compares like with like.
                item = dataclasses.replace(item, state="")
            # `_eval_item` is the harness's own per-item logic, so the accuracy this
            # script prints is the accuracy the eval config prints. The separate
            # forward pass below is only for the prediction histogram; if the two
            # ever disagree the script stops rather than describe a different model.
            result = _eval_item(model, item)
            if result is None:
                continue
            _, is_correct, _ = result
            predicted = _predict(model, item)
            gold = _gold(item)
            if (predicted == gold) != is_correct:
                disagreements += 1
            key = (item.group, item.question.get("type"))
            correct[key][1] += 1
            correct[key][0] += int(predicted == gold)
            preds[key][predicted] += 1
            if args.dump_preds:
                records.append(
                    {
                        "id": item.id,
                        "group": item.group,
                        "type": item.question.get("type"),
                        "gold": str(gold),
                        "pred": str(predicted),
                        "correct": bool(predicted == gold),
                    }
                )

    if disagreements:
        raise SystemExit(
            f"{disagreements} item(s) where the grouped prediction disagrees with "
            "model.training.supervised._eval_item; the grouped table would not "
            "describe the score the harness reports"
        )

    print(f"\n=== {args.load_heads} (choice_mode={choice_mode}) ===")
    rows = []
    total_ok = total_n = 0
    for key in sorted(correct, key=lambda k: (str(k[1]), str(k[0]))):
        ok, n = correct[key]
        total_ok += ok
        total_n += n
        rows.append(
            {
                "group": key[0],
                "type": key[1],
                "correct": ok,
                "n": n,
                "accuracy": ok / n,
                "predictions": {str(k): v for k, v in preds[key].items()},
            }
        )
        print(
            f"  {str(key[0]):<12s} {str(key[1]):<7s} {ok:>3d}/{n:<3d} = {ok / n:.4f}"
            f"   preds={dict(preds[key])}"
        )
    print(f"  {'TOTAL':<12s} {'':<7s} {total_ok:>3d}/{total_n:<3d} = {total_ok / max(total_n, 1):.4f}")

    if args.dump_preds:
        import json

        args.dump_preds.parent.mkdir(parents=True, exist_ok=True)
        args.dump_preds.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        print(f"Predictions written to {args.dump_preds}")

    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.load_heads),
                    "config": str(args.config),
                    "source": args.source,
                    "choice_mode": choice_mode,
                    "total": {"correct": total_ok, "n": total_n},
                    "by_group": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Report saved to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
