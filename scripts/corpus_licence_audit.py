#!/usr/bin/env python3
"""Count the canonical corpus by licence group and compute exclusion scenarios.

The licence verdicts and the evidence for them live in
``research/30-canonical-source-licences.md``.  This script reproduces the counts
and the two exclusion scenarios.  It does not modify data.

Usage:
    python scripts/corpus_licence_audit.py \
        --root data/benchmarks --typed-decisions /path/typed_decisions.jsonl \
        --repo-root . --json /tmp/licence-audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CANONICAL_FILES = (
    ("target_30k", "label_target_30k_jev.jsonl"),
    ("residual_20k", "residual_train_20000_jev.jsonl"),
    ("typed_decisions", None),
    ("apigen", "apigen_policy_compliance.jsonl"),
    ("apigen", "apigen_tool_action.jsonl"),
    ("apigen", "apigen_tool_argument.jsonl"),
    ("apigen", "apigen_tool_choice.jsonl"),
    ("apigen", "apigen_tool_next_action.jsonl"),
    ("abstention", "abstention_train_20k.jsonl"),
    ("abstention", "abstention_train_30k.jsonl"),
    ("mmlu_pro", "mmlu_pro_short_reasoning_choice.jsonl"),
    ("long_context", "hotpotqa_long_choice.jsonl"),
    ("long_context", "musique_long_choice.jsonl"),
    ("long_context", "longbench_long_choice.jsonl"),
    ("long_context", "twowiki_long_choice.jsonl"),
    ("race_short", "race_short_passage_choice.jsonl"),
)

# Synthetic domain -> seed dataset.  A seeded item inherits the seed's terms.
SEED_SOURCE = {
    "medical_triage": "mednli",
    "legal_judgment": "contractnli",
    "code_review": "codesearchnet",
    "financial_analysis": "tabfact",
    "scientific_reasoning": "arc",
    "content_analysis": "fever",
    "academic_reasoning": "mmlu",
    "commonsense_decision": "commonsenseqa",
    "logical_inference": "logiqa",
    "adversarial_inference": "anli",
    "passage_decision": "boolq",
    "long_document": "quality",
    "multi_hop_reasoning": "hotpotqa",
    "numerical_reasoning": "drop",
}
PURE_GENERATION = {
    "product_categorization",
    "resource_management",
    "sequential_action",
    "safety_moderation",
    "navigation_planning",
    "spatial_reasoning",
    "multi_agent_coordination",
    "education_assessment",
    "game_strategy",
}

# group -> (verdict, evidence URL)
VERDICTS = {
    "anli": ("restrictive", "https://huggingface.co/datasets/facebook/anli"),
    "contractnli": ("restrictive", "https://huggingface.co/datasets/kiddothe2b/contract-nli"),
    "apigen": ("restrictive", "https://huggingface.co/datasets/Salesforce/APIGen-MT-5k"),
    "race": ("restrictive", "https://huggingface.co/datasets/ehovy/race"),
    "agnews": ("restrictive", "https://huggingface.co/datasets/fancyzhx/ag_news"),
    "mednli": ("restrictive", "https://physionet.org/content/mednli/1.0.0/"),
    "yelp": ("restrictive", "https://business.yelp.com/data/resources/open-dataset/"),
    "multirc": ("restrictive", "https://huggingface.co/datasets/aps/super_glue"),
    "hellaswag": ("unknown", "https://huggingface.co/datasets/Rowan/hellaswag"),
    "sst2": ("unknown", "https://huggingface.co/datasets/stanfordnlp/sst2"),
    "sst5": ("unknown", "https://huggingface.co/datasets/SetFit/sst5"),
    "swag": ("unknown", "https://huggingface.co/datasets/allenai/swag"),
    "piqa": ("unknown", "https://huggingface.co/datasets/ybisk/piqa"),
    "logiqa": ("unknown", "https://huggingface.co/datasets/lucasmccabe/logiqa"),
    "codesearchnet": ("unknown", "https://huggingface.co/datasets/code-search-net/code_search_net"),
    "quality": ("unknown", "https://github.com/emozilla/quality"),
    "2wiki": ("unknown", "https://github.com/Alab-NLP/2WikiMultiHopQA"),
    "stsb": ("unknown", "https://huggingface.co/datasets/mteb/stsbenchmark-sts"),
    "mnli": ("permissive", "https://huggingface.co/datasets/nyu-mll/multi_nli"),
    "mmlu": ("permissive", "https://huggingface.co/datasets/cais/mmlu"),
    "mmlu_pro": ("permissive", "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"),
    "commonsenseqa": ("permissive", "https://huggingface.co/datasets/tau/commonsense_qa"),
    "boolq": ("permissive", "https://huggingface.co/datasets/google/boolq"),
    "fever": ("permissive", "https://huggingface.co/datasets/copenlu/fever_gold_evidence"),
    "tabfact": ("permissive", "https://huggingface.co/datasets/wenhu/tab_fact"),
    "banking77": ("permissive", "https://huggingface.co/datasets/legacy-datasets/banking77"),
    "arc": ("permissive", "https://huggingface.co/datasets/allenai/ai2_arc"),
    "hotpotqa": ("permissive", "https://huggingface.co/datasets/hotpotqa/hotpot_qa"),
    "musique": ("permissive", "https://github.com/StonyBrookNLP/musique/blob/main/LICENSE"),
    "longbench": ("permissive", "https://github.com/THUDM/LongBench/blob/main/LICENSE"),
    "drop": ("permissive", "https://huggingface.co/datasets/ucinlp/drop"),
    "winogrande": ("permissive", "https://github.com/allenai/winogrande/blob/master/LICENSE"),
    "typed_decisions": ("permissive", "https://huggingface.co/datasets/LocalLLaMA/typed-decisions"),
    "openjev_authored144": ("permissive", "scripts/build_authored_holdout.py"),
    "project": ("permissive", "data/synthetic.py"),
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def licence_group(item: dict) -> str:
    source = str(item.get("source") or "missing")
    item_id = str(item.get("id") or "")
    if source == "synthetic":
        parts = item_id.split("-")
        domain = parts[1] if len(parts) > 1 else "?"
        return SEED_SOURCE.get(domain, "project" if domain in PURE_GENERATION else "synthetic_unknown")
    if source.startswith("apigen"):
        return "apigen"
    if source in ("race", "race_short_passage_choice"):
        return "race"
    if source == "hotpotqa_long_choice":
        return "musique" if item_id.startswith("musique-long-") else "hotpotqa"
    if source == "longbench_long_choice":
        return "longbench"
    if source == "2wiki_long_choice":
        return "2wiki"
    if source == "mmlu_pro_short_reasoning_choice":
        return "mmlu_pro"
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--typed-decisions", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data.synthetic import family_to_typed_questions

    root = args.root.resolve()
    synth_dir = args.synthetic_dir.resolve() if args.synthetic_dir else root / "synthetic"
    typed_path = args.typed_decisions or (root / "typed_decisions.jsonl")

    seen: set[Any] = set()
    groups: dict[str, Counter] = defaultdict(Counter)
    by_corpus: dict[str, Counter] = defaultdict(Counter)

    def add(item: dict, corpus: str) -> bool:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return False
        if corpus == "race_short":
            key: Any = (item_id, json.dumps(item.get("question") or {}, sort_keys=True))
        else:
            key = item_id
        if key in seen:
            return False
        seen.add(key)
        q_type = str((item.get("question") or {}).get("type") or "missing")
        group = licence_group(item)
        groups[group][q_type] += 1
        by_corpus[corpus][q_type] += 1
        return True

    missing: list[str] = []
    for corpus, name in CANONICAL_FILES:
        path = typed_path if name is None else root / name
        if not path.exists():
            missing.append(str(path))
            continue
        for item in read_jsonl(path):
            add(item, corpus)
        if corpus == "typed_decisions":
            for families_path in sorted(synth_dir.glob("*_families.jsonl")):
                domain = families_path.name[: -len("_families.jsonl")]
                variants_path = synth_dir / f"{domain}_variants.jsonl"
                families = list(read_jsonl(families_path))
                variants = list(read_jsonl(variants_path)) if variants_path.exists() else []
                for index, family in enumerate(families):
                    variant_doc = variants[index] if index < len(variants) else {}
                    for item in family_to_typed_questions(
                        family, variant_doc, domain, family.get("_family_idx", index)
                    ):
                        add(item, "synthetic")
    if missing:
        raise SystemExit("canonical files missing:\n  " + "\n  ".join(missing))

    total = sum(sum(counter.values()) for counter in groups.values())
    current = Counter()
    for counter in groups.values():
        current.update(counter)

    def scenario(kinds: set[str]) -> dict[str, Any]:
        excluded = Counter()
        for group, counter in groups.items():
            verdict = VERDICTS.get(group, ("unknown", ""))[0]
            if verdict in kinds:
                excluded.update(counter)
        return {
            "excluded": sum(excluded.values()),
            "remaining": total - sum(excluded.values()),
            "excluded_by_type": dict(excluded),
            "remaining_by_type": {
                kind: current.get(kind, 0) - excluded.get(kind, 0)
                for kind in ("noul", "choice", "score")
            },
        }

    report = {
        "total": total,
        "by_type": dict(current),
        "groups": {
            group: {
                "records": sum(counter.values()),
                "by_type": dict(counter),
                "verdict": VERDICTS.get(group, ("unknown", ""))[0],
                "evidence": VERDICTS.get(group, ("unknown", ""))[1],
            }
            for group, counter in sorted(groups.items(), key=lambda kv: -sum(kv[1].values()))
        },
        "scenarios": {
            "restrictive": scenario({"restrictive"}),
            "restrictive_and_unknown": scenario({"restrictive", "unknown"}),
        },
        "unclassified_groups": sorted(set(groups) - set(VERDICTS)),
    }

    text = json.dumps(report, indent=2, sort_keys=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
