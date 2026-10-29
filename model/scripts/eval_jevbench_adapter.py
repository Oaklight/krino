#!/usr/bin/env python3
"""Score JevBench public items with the board's own reranker mapping.

``research/13`` found our native readout below the published ``qwen3-reranker-4b``
row on the same 67 hard-choice items (0.373 and 0.493 across two seeds against
0.507). The board's adapter is published, and reading it shows why that
comparison was not like-for-like: its **query carries the complete answer
rubric**, while ours carries only the state and the question. That is an input
difference the model sees.

The rest of the adapter cannot move accuracy. Its temperature grid and its noul
threshold are both monotone maps, so they leave every argmax where it was --
the same argument that ruled out temperature scaling for the QWK clause. Only
the prompt differs.

So this script runs one checkpoint under either mapping and reports accuracy by
tier and type, which separates protocol from weights:

* ``--prompt official`` reproduces the board's mapping: rubric in the query,
  its exact instruction wording, compact JSON for structured states.
* ``--prompt ours`` uses ``model/src/reranker_prompt.py`` as the harness does.

Running the **base** model under ``--prompt official`` is the validation: if our
reimplementation lands near the published 0.495 (all types) and 0.507 (choice)
on the public hard items, the harness matches and any remaining difference is
ours to explain.

Usage:
    python model/scripts/eval_jevbench_adapter.py \
        --model Qwen/Qwen3-Reranker-4B --prompt official \
        --data-root /home/pding/projects/open-decisions --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import DATA_DIR, render_jevbench_state  # noqa: E402
from model.src.reranker_prompt import (  # noqa: E402
    INSTRUCTION,
    PREFIX,
    SUFFIX,
    build_query_doc,
    score_options_native,
)

# From jevbench/adapters/reranker.py, verbatim.
OFFICIAL_INSTRUCTIONS = {
    "choice": "Classify: which option correctly answers the question for the given situation? Rate how well each option fits.",
    "noul": "Classify: does yes or no correctly answer the question for the given situation? Rate how well each option fits.",
    "score": "Classify: which rubric level correctly rates the situation? Rate how well each level fits.",
}


def official_text(value: Any) -> str:
    """The board's rendering: compact, sorted JSON for structured states."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def official_criteria(question: dict[str, Any]) -> dict[str, str]:
    q_type = question["type"]
    criteria = question.get("criteria")
    if q_type == "score":
        return {str(index): official_text(v) for index, v in enumerate(criteria or [])}
    if isinstance(criteria, dict):
        if q_type == "noul":
            return {
                "no": official_text(criteria.get("false", criteria.get("no", "No"))),
                "yes": official_text(criteria.get("true", criteria.get("yes", "Yes"))),
            }
        return {str(k): official_text(v) for k, v in criteria.items()}
    return {}


def official_query(state: Any, question: dict[str, Any], labels: list[str]) -> str:
    """state + question + the full rubric, which is the part we were omitting."""
    rubric = official_criteria(question)
    lines = [
        "Situation:",
        official_text(state),
        "",
        "Question:",
        official_text(question.get("instructions", "")),
        "",
        "Answer rubric:",
    ]
    lines.extend(f"- {label}: {rubric.get(label, str(label))}" for label in labels)
    return "\n".join(lines)


def official_option(question: dict[str, Any], label: str, description: str) -> str:
    kind = "Rubric level" if question["type"] == "score" else "Answer option"
    return f"{kind} {label}: {description}"


def labels_and_texts(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    q_type = question["type"]
    criteria = question.get("criteria")
    if q_type == "noul":
        return ["no", "yes"], [
            official_text(criteria.get("false", criteria.get("no", "No")))
            if isinstance(criteria, dict)
            else "No",
            official_text(criteria.get("true", criteria.get("yes", "Yes")))
            if isinstance(criteria, dict)
            else "Yes",
        ]
    if q_type == "score":
        if isinstance(criteria, list):
            return [str(i) for i in range(len(criteria))], [official_text(v) for v in criteria]
        return list(criteria), [official_text(v) for v in criteria.values()]
    if isinstance(criteria, dict):
        return list(criteria), [official_text(v) for v in criteria.values()]
    return list(criteria or []), [str(v) for v in criteria or []]


def load_items() -> list[dict[str, Any]]:
    items = []
    cache = DATA_DIR / "jevbench"
    for tier in ("easy", "hard", "original"):
        path = cache / f"{tier}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    task = json.loads(line)
                    task["_tier"] = tier
                    items.append(task)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--load-heads", type=Path, default=None)
    parser.add_argument("--lora-layers", type=int, default=-1)
    parser.add_argument("--readout", default="last")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--prompt", choices=("official", "ours"), default="official")
    parser.add_argument("--tiers", default="hard")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import data.pipeline as pipeline_mod

    if args.data_root:
        pipeline_mod.DATA_DIR = args.data_root / "data" / "benchmarks"

    cfg_path = ROOT / "model" / "configs" / "r2_reranker4b.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    from model.src.gpu_config import GPUConfig, estimate_model_billions
    from model.src.model_factory import build_decision_model

    gpu = GPUConfig.auto_detect(estimate_model_billions(args.model)).apply_overrides(
        max_length=32768
    )
    if args.load_heads:
        model = build_decision_model(
            model_name=args.model,
            gpu_config=gpu,
            rank=args.rank,
            mlp_layers=0,
            noul_rank=args.rank,
            max_length=32768,
            shared_attention=True,
            readout=args.readout,
            lora_layers=args.lora_layers,
            lora_rank=16,
            lora_alpha=32,
            lora_targets=("q_proj", "v_proj"),
        )
        model.load_heads(args.load_heads)
        backbone, tokenizer = model.backbone, model.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        backbone = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        backbone.to(args.device)
    backbone.eval()

    items = [t for t in load_items() if t["_tier"] in args.tiers.split(",")]
    print(f"{len(items)} items, prompt={args.prompt}", flush=True)

    by_tier = defaultdict(lambda: [0, 0])
    by_tier_type = defaultdict(lambda: [0, 0])
    for index, task in enumerate(items, start=1):
        question = task["question"]
        q_type = question["type"]
        labels, texts = labels_and_texts(question)
        if not labels:
            continue
        expected = str(task["expected"])
        if q_type == "noul":
            expected = "yes" if str(task["expected"]).lower() in ("yes", "true") else "no"
        if q_type == "score":
            # Score items keep their numeric level as the label.
            expected = str(int(round(float(task["expected"]))))
        if expected not in labels:
            continue

        if args.prompt == "official":
            query = official_query(task["state"], question, labels)
            docs = [official_option(question, label, texts[i]) for i, label in enumerate(labels)]
            instruction = OFFICIAL_INSTRUCTIONS[q_type]
            scores = []
            for doc in docs:
                text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                pre = tokenizer.encode(PREFIX, add_special_tokens=False)
                suf = tokenizer.encode(SUFFIX, add_special_tokens=False)
                ids = tokenizer.encode(text, add_special_tokens=False)[
                    : args.max_length - len(pre) - len(suf)
                ]
                inp = torch.tensor([pre + ids + suf], device=backbone.device)
                with torch.no_grad():
                    logits = backbone(
                        input_ids=inp, attention_mask=torch.ones_like(inp)
                    ).logits[0, -1, :].float()
                true_id = tokenizer.convert_tokens_to_ids("yes")
                false_id = tokenizer.convert_tokens_to_ids("no")
                pair = torch.softmax(torch.stack([logits[false_id], logits[true_id]]), 0)
                p = float(pair[1])
                p = min(1 - 1e-6, max(1e-6, p))
                scores.append(math.log(p) - math.log(1 - p))
        else:
            state = render_jevbench_state(task["state"])
            scores = score_options_native(
                backbone,
                tokenizer,
                state,
                question,
                list(zip(labels, texts)),
                max_length=args.max_length,
                device=args.device,
            )
            scores = [
                math.log(max(s, 1e-6)) - math.log(max(1 - s, 1e-6)) for s in scores
            ]

        pick = labels[max(range(len(scores)), key=lambda i: (scores[i], -i))]
        correct = pick == expected
        by_tier[task["_tier"]][0] += correct
        by_tier[task["_tier"]][1] += 1
        by_tier_type[(task["_tier"], q_type)][0] += correct
        by_tier_type[(task["_tier"], q_type)][1] += 1
        if index % 25 == 0:
            print(f"  {index}/{len(items)}", flush=True)

    report = {"prompt": args.prompt, "model": args.model, "by_tier": {}, "by_tier_type": {}}
    for tier, (c, n) in sorted(by_tier.items()):
        report["by_tier"][tier] = {"correct": c, "n": n, "accuracy": c / n}
        print(f"  {tier:<10} {c}/{n} = {c / n:.4f}", flush=True)
    for (tier, q_type), (c, n) in sorted(by_tier_type.items()):
        report["by_tier_type"][f"{tier}/{q_type}"] = {"correct": c, "n": n, "accuracy": c / n}
        print(f"  {tier}/{q_type:<6} {c}/{n} = {c / n:.4f}", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report saved to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
