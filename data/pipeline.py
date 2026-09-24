"""Data pipeline: download, convert, and split public benchmarks into typed-question JSONL."""

from __future__ import annotations

import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from .format import TypedQuestion

DATA_DIR = Path(__file__).resolve().parent / "benchmarks"


def _download_parquet(url: str, cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        cache_path.write_bytes(resp.read())
    print(f"  downloaded {cache_path.name} ({cache_path.stat().st_size // 1024} KB)", file=sys.stderr)
    return cache_path


def _read_parquet(path: Path) -> "pyarrow.Table":
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for data preparation: pip install 'jev-explore[data]'")
    return pq.read_table(path)


def _load_hf_parquet(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    """Download and read a HuggingFace dataset split as parquet.

    Supports multi-shard datasets by fetching 0000.parquet, 0001.parquet, ...
    until a 404 is returned.
    """
    safe_name = dataset.replace("/", "__")
    base_url = f"https://huggingface.co/datasets/{dataset}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}"
    cache_dir = DATA_DIR / safe_name

    # Migrate old single-shard cache filename to new naming convention
    old_cache = cache_dir / f"{config}_{split}.parquet"
    new_first = cache_dir / f"{config}_{split}_0000.parquet"
    if old_cache.exists() and not new_first.exists():
        try:
            old_cache.rename(new_first)
        except FileNotFoundError:
            pass

    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError("pyarrow is required for data preparation: pip install 'jev-explore[data]'")

    MAX_SHARDS = 1000
    tables: list[pa.Table] = []
    for shard_idx in range(MAX_SHARDS):
        shard_name = f"{config}_{split}_{shard_idx:04d}.parquet"
        cache_path = cache_dir / shard_name
        url = f"{base_url}/{shard_idx:04d}.parquet"

        if not cache_path.exists():
            try:
                _download_parquet(url, cache_path)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    if shard_idx == 0:
                        raise FileNotFoundError(
                            f"No parquet shards found for {dataset}/{config}/{split}"
                        )
                    break
                raise

        tables.append(_read_parquet(cache_path))
    else:
        print(f"  warning: hit {MAX_SHARDS}-shard cap for {dataset}/{config}/{split}", file=sys.stderr)

    return pa.concat_tables(tables).to_pylist()


# --- Banking77 ---

BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
    "change_pin", "compromised_card", "contactless_not_working",
    "country_support", "declined_card_payment", "declined_cash_withdrawal",
    "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "freeze_card", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "signing_up", "spare_card", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
    "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def load_banking77() -> Iterator[TypedQuestion]:
    criteria = {label: label.replace("_", " ") for label in BANKING77_LABELS}
    for split_name in ("train", "test"):
        rows = _load_hf_parquet("legacy-datasets/banking77", "default", split_name)
        for i, row in enumerate(rows):
            text = row.get("text", "")
            label_idx = row.get("label", 0)
            label = BANKING77_LABELS[label_idx] if isinstance(label_idx, int) and label_idx < len(BANKING77_LABELS) else str(label_idx)
            yield TypedQuestion.choice(
                id=f"banking77-{split_name}-{i:05d}",
                state=text,
                instructions="Which banking intent category does this customer message belong to?",
                criteria=criteria,
                label=label,
                source="banking77",
                split=split_name,
                group=f"banking77-{i % 100}",
            )


# --- SST-2 ---

def load_sst2() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("stanfordnlp/sst2", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            sentence = row.get("sentence", "")
            label_val = row.get("label", 0)
            yield TypedQuestion.noul(
                id=f"sst2-{out_split}-{i:05d}",
                state=sentence,
                instructions="Is the sentiment of this sentence positive?",
                label=bool(label_val),
                source="sst2",
                split=out_split,
                group=f"sst2-{i % 50}",
            )


# --- AG News ---

AGNEWS_LABELS = {"0": "world", "1": "sports", "2": "business", "3": "science_technology"}
AGNEWS_CRITERIA = {
    "world": "World news and international affairs",
    "sports": "Sports news and athletics",
    "business": "Business and financial news",
    "science_technology": "Science and technology news",
}


def load_agnews() -> Iterator[TypedQuestion]:
    for split_name in ("train", "test"):
        rows = _load_hf_parquet("fancyzhx/ag_news", "default", split_name)
        for i, row in enumerate(rows):
            text = row.get("text", "")
            label_idx = row.get("label", 0)
            label = AGNEWS_LABELS.get(str(label_idx), str(label_idx))
            yield TypedQuestion.choice(
                id=f"agnews-{split_name}-{i:05d}",
                state=text,
                instructions="Which news category does this article belong to?",
                criteria=AGNEWS_CRITERIA,
                label=label,
                source="agnews",
                split=split_name,
                group=f"agnews-{i % 200}",
            )


# --- MNLI ---

def load_mnli() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation_matched"):
        rows = _load_hf_parquet("nyu-mll/multi_nli", "default", split_name)
        out_split = "test" if "validation" in split_name else "train"
        for i, row in enumerate(rows):
            premise = row.get("premise", "")
            hypothesis = row.get("hypothesis", "")
            label_idx = row.get("label", -1)
            if label_idx == -1:
                continue
            state = f"Premise: {premise}\nHypothesis: {hypothesis}"
            yield TypedQuestion.noul(
                id=f"mnli-{out_split}-{i:05d}",
                state=state,
                instructions="Does the premise entail the hypothesis?",
                label=(label_idx == 0),
                source="mnli",
                split=out_split,
                group=row.get("genre", f"mnli-{i % 50}"),
            )


# --- Typed-Decisions (LocalLLaMA community benchmark) ---

def load_typed_decisions() -> Iterator[TypedQuestion]:
    for split_name in ("train", "test"):
        rows = _load_hf_parquet("LocalLLaMA/typed-decisions", "all", split_name)
        for i, row in enumerate(rows):
            state = row.get("state", "")
            if isinstance(state, str) and state.startswith("{"):
                state = json.loads(state)
            questions = row.get("questions", {})
            if isinstance(questions, str):
                questions = json.loads(questions) if questions.startswith("{") else {}
            gold = row.get("gold", {})
            if isinstance(gold, str):
                gold = json.loads(gold) if gold.startswith("{") else {}
            workflow = row.get("workflow", f"td-{i % 20}")
            for qid, q_def in questions.items():
                if not isinstance(q_def, dict) or "type" not in q_def:
                    continue
                gold_answer = gold.get(qid, {})
                label = gold_answer.get("label", None)
                if label is None:
                    continue
                yield TypedQuestion(
                    id=f"typed_decisions-{split_name}-{i:05d}-{qid}",
                    state=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                    question=q_def,
                    label=label,
                    source="typed_decisions",
                    split=split_name,
                    group=workflow,
                )


# --- STS-B (Semantic Textual Similarity) ---

STS_LEVELS = [
    "No similarity", "Slight similarity", "Moderate similarity",
    "Good similarity", "Strong similarity", "Perfect similarity",
]


def load_stsb() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation", "test"):
        rows = _load_hf_parquet("nyu-mll/glue", "stsb", split_name)
        out_split = "test" if split_name in ("validation", "test") else "train"
        for i, row in enumerate(rows):
            sentence1 = row.get("sentence1", "")
            sentence2 = row.get("sentence2", "")
            score = float(row.get("label", 0))
            label = float(min(5, max(0, round(score))))
            state = f"Sentence A: {sentence1}\nSentence B: {sentence2}"
            yield TypedQuestion.score(
                id=f"stsb-{split_name}-{i:05d}",
                state=state,
                instructions="How semantically similar are these two sentences?",
                criteria=STS_LEVELS,
                label=label,
                source="stsb",
                split=out_split,
                group=f"stsb-{i % 50}",
            )


# --- SST-5 (Fine-grained Sentiment) ---

SST5_LEVELS = [
    "Very negative", "Negative", "Neutral", "Positive", "Very positive",
]


def load_sst5() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation", "test"):
        rows = _load_hf_parquet("SetFit/sst5", "default", split_name)
        out_split = "test" if split_name in ("validation", "test") else "train"
        for i, row in enumerate(rows):
            text = row.get("text", "")
            label_val = row.get("label", 0)
            yield TypedQuestion.score(
                id=f"sst5-{split_name}-{i:05d}",
                state=text,
                instructions="What is the sentiment of this sentence?",
                criteria=SST5_LEVELS,
                label=float(label_val),
                source="sst5",
                split=out_split,
                group=f"sst5-{i % 50}",
            )


# --- ARC (AI2 Reasoning Challenge) ---


def load_arc() -> Iterator[TypedQuestion]:
    for config in ("ARC-Easy", "ARC-Challenge"):
        for split_name in ("train", "test", "validation"):
            rows = _load_hf_parquet("allenai/ai2_arc", config, split_name)
            out_split = "test" if split_name == "validation" else split_name
            for i, row in enumerate(rows):
                choices = row.get("choices", {})
                texts = choices.get("text", [])
                labels = choices.get("label", [])
                criteria = {lbl: txt for lbl, txt in zip(labels, texts)}
                answer_key = row.get("answerKey", "")
                if answer_key not in criteria:
                    continue
                yield TypedQuestion.choice(
                    id=f"arc-{config.lower()}-{split_name}-{i:05d}",
                    state=row.get("question", ""),
                    instructions="Which answer is correct?",
                    criteria=criteria,
                    label=answer_key,
                    source="arc",
                    split=out_split,
                    group=f"arc-{config.lower()}",
                )


# --- RACE (Reading Comprehension) ---

RACE_KEYS = ["A", "B", "C", "D"]


def load_race() -> Iterator[TypedQuestion]:
    for config in ("middle", "high"):
        for split_name in ("train", "test", "validation"):
            rows = _load_hf_parquet("ehovy/race", config, split_name)
            out_split = "test" if split_name == "validation" else split_name
            for i, row in enumerate(rows):
                article = row.get("article", "")
                question = row.get("question", "")
                options = row.get("options", [])
                answer = row.get("answer", "")
                if len(options) != 4 or answer not in RACE_KEYS:
                    continue
                criteria = {k: opt for k, opt in zip(RACE_KEYS, options)}
                yield TypedQuestion.choice(
                    id=f"race-{config}-{split_name}-{i:05d}",
                    state=article,
                    instructions=f"Based on the passage above, answer: {question}",
                    criteria=criteria,
                    label=answer,
                    source="race",
                    split=out_split,
                    group=f"race-{config}",
                )


# --- HellaSwag (Commonsense Completion) ---

HELLASWAG_KEYS = {"0", "1", "2", "3"}


def load_hellaswag() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("Rowan/hellaswag", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            label = str(row.get("label", ""))
            if label not in HELLASWAG_KEYS:
                continue
            ctx = row.get("ctx", "")
            activity = row.get("activity_label", "")
            endings = row.get("endings", [])
            if len(endings) != 4:
                continue
            state = f"{activity}: {ctx}" if activity else ctx
            criteria = {str(j): e for j, e in enumerate(endings)}
            yield TypedQuestion.choice(
                id=f"hellaswag-{split_name}-{i:05d}",
                state=state,
                instructions="Which ending most naturally completes the context?",
                criteria=criteria,
                label=label,
                source="hellaswag",
                split=out_split,
                group=row.get("activity_label") or f"hellaswag-{i % 100}",
            )


# --- TabFact (Table Verification) ---


def load_tabfact() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation", "test"):
        rows = _load_hf_parquet("wenhu/tab_fact", "tab_fact", split_name)
        out_split = "test" if split_name in ("validation", "test") else "train"
        for i, row in enumerate(rows):
            table_text = row.get("table_text", "")
            caption = row.get("table_caption", "")
            statement = row.get("statement", "")
            label_val = row.get("label", 0)
            state = f"Table: {caption}\n{table_text}\n\nStatement: {statement}"
            yield TypedQuestion.noul(
                id=f"tabfact-{split_name}-{i:05d}",
                state=state,
                instructions="Is the statement entailed by the table?",
                label=bool(label_val),
                source="tabfact",
                split=out_split,
                group=row.get("table_id", f"tabfact-{i % 200}"),
            )


# --- FEVER (Fact Verification) ---

FEVER_CRITERIA = {
    "supports": "The evidence supports the claim",
    "refutes": "The evidence refutes or contradicts the claim",
    "nei": "There is not enough information to verify the claim",
}
FEVER_LABEL_MAP = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "nei",
}


def load_fever() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("copenlu/fever_gold_evidence", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            claim = row.get("claim", "")
            raw_label = row.get("label", "")
            label = FEVER_LABEL_MAP.get(raw_label)
            if label is None:
                continue
            evidence_list = row.get("evidence", [])
            evidence_texts = []
            # copenlu/fever_gold_evidence format: [page_title, sentence_text]
            for ev in evidence_list:
                if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                    evidence_texts.append(str(ev[-1]))
            evidence_str = "\n".join(evidence_texts) if evidence_texts else "(no evidence)"
            state = f"Claim: {claim}\n\nEvidence:\n{evidence_str}"
            yield TypedQuestion.choice(
                id=f"fever-{split_name}-{i:05d}",
                state=state,
                instructions="Based on the evidence, what is the verdict on this claim?",
                criteria=FEVER_CRITERIA,
                label=label,
                source="fever",
                split=out_split,
                group=f"fever-{i % 200}",
            )


# --- Yelp Reviews (Fine-grained Sentiment, Score) ---

YELP_LEVELS = [
    "1 star — Terrible",
    "2 stars — Poor",
    "3 stars — Average",
    "4 stars — Good",
    "5 stars — Excellent",
]

_YELP_TRAIN_SAMPLE = 20_000


def load_yelp() -> Iterator[TypedQuestion]:
    rng = random.Random(42)
    for split_name in ("train", "test"):
        rows = _load_hf_parquet("Yelp/yelp_review_full", "yelp_review_full", split_name)
        if split_name == "train" and len(rows) > _YELP_TRAIN_SAMPLE:
            rows = rng.sample(rows, _YELP_TRAIN_SAMPLE)
        for i, row in enumerate(rows):
            text = row.get("text", "")
            if not text:
                continue
            label_val = row.get("label", 0)
            yield TypedQuestion.score(
                id=f"yelp-{split_name}-{i:05d}",
                state=text,
                instructions="What star rating does this review correspond to?",
                criteria=YELP_LEVELS,
                label=float(label_val),
                source="yelp",
                split=split_name,
                group=f"yelp-{i % 200}",
            )


# --- SWAG (Grounded Commonsense, 4-way Choice) ---

SWAG_KEYS = ["A", "B", "C", "D"]


def load_swag() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("allenai/swag", "regular", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            label_idx = row.get("label", -1)
            if not isinstance(label_idx, int) or label_idx < 0 or label_idx > 3:
                continue
            startphrase = row.get("startphrase", "")
            endings = [row.get(f"ending{j}", "") for j in range(4)]
            if not startphrase or not all(endings):
                continue
            criteria = {k: e for k, e in zip(SWAG_KEYS, endings)}
            yield TypedQuestion.choice(
                id=f"swag-{split_name}-{i:05d}",
                state=startphrase,
                instructions="Which ending most naturally completes the sentence?",
                criteria=criteria,
                label=SWAG_KEYS[label_idx],
                source="swag",
                split=out_split,
                group=row.get("video-id", f"swag-{i % 100}"),
            )


# --- MultiRC (Multi-Sentence Reading Comprehension, Noul) ---


def load_multirc() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("aps/super_glue", "multirc", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            label_val = row.get("label", -1)
            if label_val not in (0, 1):
                continue
            paragraph = row.get("paragraph", "")
            question = row.get("question", "")
            answer = row.get("answer", "")
            if not paragraph or not question or not answer:
                continue
            idx = row.get("idx", {})
            para_idx = idx.get("paragraph", i % 50) if isinstance(idx, dict) else i % 50
            state = f"Passage: {paragraph}\n\nQuestion: {question}\n\nAnswer: {answer}"
            yield TypedQuestion.noul(
                id=f"multirc-{split_name}-{i:05d}",
                state=state,
                instructions="Is this answer correct for the given question and passage?",
                label=bool(label_val),
                source="multirc",
                split=out_split,
                group=f"multirc-p{para_idx}",
            )


# --- MedNLI (Medical NLI, Noul) ---

MEDNLI_LABEL_MAP = {
    "entailment": True,
    "contradiction": False,
    "neutral": False,
}


def load_mednli() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation", "test"):
        rows = _load_hf_parquet("presencesw/mednli", "default", split_name)
        out_split = "test" if split_name in ("validation", "test") else "train"
        for i, row in enumerate(rows):
            gold_label = row.get("gold_label", "")
            label = MEDNLI_LABEL_MAP.get(gold_label)
            if label is None:
                continue
            sentence1 = row.get("sentence1", "")
            sentence2 = row.get("sentence2", "")
            if not sentence1 or not sentence2:
                continue
            state = f"Premise: {sentence1}\nHypothesis: {sentence2}"
            yield TypedQuestion.noul(
                id=f"mednli-{split_name}-{i:05d}",
                state=state,
                instructions="Does the premise entail the hypothesis?",
                label=label,
                source="mednli",
                split=out_split,
                group=f"mednli-{i % 50}",
            )


# --- ContractNLI (Legal Clause Entailment, Noul) ---

# 0=NotMentioned, 1=Entailment, 2=Contradiction
CONTRACTNLI_LABEL_MAP = {0: False, 1: True, 2: False}


def load_contractnli() -> Iterator[TypedQuestion]:
    for config in ("contractnli_a", "contractnli_b"):
        for split_name in ("train", "validation", "test"):
            rows = _load_hf_parquet("kiddothe2b/contract-nli", config, split_name)
            out_split = "test" if split_name in ("validation", "test") else "train"
            for i, row in enumerate(rows):
                label_idx = row.get("label", -1)
                label = CONTRACTNLI_LABEL_MAP.get(label_idx)
                if label is None:
                    continue
                premise = row.get("premise", "")
                hypothesis = row.get("hypothesis", "")
                if not premise or not hypothesis:
                    continue
                state = f"Contract clause: {premise}\n\nProposition: {hypothesis}"
                yield TypedQuestion.noul(
                    id=f"contractnli-{config}-{split_name}-{i:05d}",
                    state=state,
                    instructions="Does the contract clause entail the proposition?",
                    label=label,
                    source="contractnli",
                    split=out_split,
                    group=f"contractnli-{config}",
                )


# --- CodeSearchNet (Code Retrieval, Choice) ---

_CSN_LANGUAGES = ("python", "java", "javascript")
_CSN_TRAIN_SAMPLE = 3_000
_CSN_KEYS = ["A", "B", "C", "D"]


def load_codesearchnet() -> Iterator[TypedQuestion]:
    rng = random.Random(42)
    for lang in _CSN_LANGUAGES:
        for split_name in ("train", "validation"):
            rows = _load_hf_parquet("code-search-net/code_search_net", lang, split_name)
            out_split = "test" if split_name == "validation" else "train"

            valid = [
                r for r in rows
                if r.get("func_documentation_string", "").strip()
                and r.get("func_code_string", "").strip()
            ]
            if len(valid) < 4:
                continue
            if split_name == "train" and len(valid) > _CSN_TRAIN_SAMPLE:
                valid = rng.sample(valid, _CSN_TRAIN_SAMPLE)

            for i, row in enumerate(valid):
                doc = row["func_documentation_string"].strip()
                correct_code = row["func_code_string"].strip()

                pool = [
                    j for j in range(len(valid))
                    if j != i and valid[j]["func_code_string"].strip() != correct_code
                ]
                if len(pool) < 3:
                    continue
                distractor_indices = rng.sample(pool, 3)
                distractors = [valid[j]["func_code_string"].strip() for j in distractor_indices]

                correct_pos = rng.randrange(4)
                options = distractors[:correct_pos] + [correct_code] + distractors[correct_pos:]

                criteria = {_CSN_KEYS[j]: opt for j, opt in enumerate(options)}
                yield TypedQuestion.choice(
                    id=f"codesearchnet-{lang}-{split_name}-{i:05d}",
                    state=f"Docstring: {doc}",
                    instructions=f"Which {lang} code snippet correctly implements the described functionality?",
                    criteria=criteria,
                    label=_CSN_KEYS[correct_pos],
                    source="codesearchnet",
                    split=out_split,
                    group=f"codesearchnet-{lang}",
                )


# --- MMLU (Massive Multitask Language Understanding) ---

MMLU_KEYS = ["A", "B", "C", "D"]


def load_mmlu() -> Iterator[TypedQuestion]:
    for split_name in ("test", "validation"):
        rows = _load_hf_parquet("cais/mmlu", "all", split_name)
        out_split = "test" if split_name == "validation" else split_name
        for i, row in enumerate(rows):
            question_text = row.get("question", "")
            choices = row.get("choices", [])
            answer_idx = row.get("answer", 0)
            subject = row.get("subject", "")
            if len(choices) != 4:
                continue
            criteria = {k: c for k, c in zip(MMLU_KEYS, choices)}
            label = MMLU_KEYS[answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < 4 else None
            if label is None:
                continue
            yield TypedQuestion.choice(
                id=f"mmlu-{split_name}-{i:05d}",
                state=question_text,
                instructions=f"Answer this {subject.replace('_', ' ')} question.",
                criteria=criteria,
                label=label,
                source="mmlu",
                split=out_split,
                group=f"mmlu-{subject}",
            )


# --- WinoGrande (Commonsense Pronoun Resolution) ---

WINOGRANDE_KEYS = ["1", "2"]


def load_winogrande() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("allenai/winogrande", "winogrande_xl", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            sentence = row.get("sentence", "")
            option1 = row.get("option1", "")
            option2 = row.get("option2", "")
            answer = str(row.get("answer", ""))
            if answer not in ("1", "2") or not sentence:
                continue
            criteria = {"1": option1, "2": option2}
            yield TypedQuestion.choice(
                id=f"winogrande-{split_name}-{i:05d}",
                state=sentence,
                instructions="Which option best fills the blank in the sentence?",
                criteria=criteria,
                label=answer,
                source="winogrande",
                split=out_split,
                group=f"winogrande-{i % 100}",
            )


# --- PIQA (Physical Intuition QA) ---

PIQA_KEYS = ["A", "B"]


def load_piqa() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("ybisk/piqa", "plain_text", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            goal = row.get("goal", "")
            sol1 = row.get("sol1", "")
            sol2 = row.get("sol2", "")
            label_idx = row.get("label", -1)
            if not isinstance(label_idx, int) or label_idx not in (0, 1):
                continue
            if not goal or not sol1 or not sol2:
                continue
            criteria = {"A": sol1, "B": sol2}
            yield TypedQuestion.choice(
                id=f"piqa-{split_name}-{i:05d}",
                state=goal,
                instructions="Which solution best achieves the goal?",
                criteria=criteria,
                label=PIQA_KEYS[label_idx],
                source="piqa",
                split=out_split,
                group=f"piqa-{i % 100}",
            )


# --- CommonsenseQA (5-way Commonsense Reasoning) ---


def load_commonsenseqa() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("tau/commonsense_qa", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            question_text = row.get("question", "")
            choices = row.get("choices", {})
            answer_key = row.get("answerKey", "")
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if len(labels) != len(texts) or len(labels) == 0:
                continue
            criteria = {lbl: txt for lbl, txt in zip(labels, texts)}
            if answer_key not in criteria:
                continue
            yield TypedQuestion.choice(
                id=f"commonsenseqa-{split_name}-{i:05d}",
                state=question_text,
                instructions="Answer this commonsense reasoning question.",
                criteria=criteria,
                label=answer_key,
                source="commonsenseqa",
                split=out_split,
                group=f"commonsenseqa-{i % 100}",
            )


# --- LogiQA (Logical Reasoning) ---

LOGIQA_KEYS = ["A", "B", "C", "D"]


def load_logiqa() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation", "test"):
        rows = _load_hf_parquet("lucasmccabe/logiqa", "default", split_name)
        out_split = "test" if split_name in ("validation", "test") else "train"
        for i, row in enumerate(rows):
            context = row.get("context", "")
            query = row.get("query", "")
            options = row.get("options", [])
            correct_option = row.get("correct_option", -1)
            if len(options) != 4:
                continue
            if not isinstance(correct_option, int) or not (0 <= correct_option < 4):
                continue
            criteria = {k: opt for k, opt in zip(LOGIQA_KEYS, options)}
            state = f"{context}\n\nQuestion: {query}" if context else query
            yield TypedQuestion.choice(
                id=f"logiqa-{split_name}-{i:05d}",
                state=state,
                instructions="Which answer is correct based on logical reasoning?",
                criteria=criteria,
                label=LOGIQA_KEYS[correct_option],
                source="logiqa",
                split=out_split,
                group=f"logiqa-{i % 100}",
            )


# --- ANLI (Adversarial NLI) ---


def load_anli() -> Iterator[TypedQuestion]:
    for round_num in (1, 2, 3):
        for split_prefix in ("train", "dev"):
            split_name = f"{split_prefix}_r{round_num}"
            out_split = "test" if split_prefix == "dev" else "train"
            rows = _load_hf_parquet("facebook/anli", "plain_text", split_name)
            for i, row in enumerate(rows):
                premise = row.get("premise", "")
                hypothesis = row.get("hypothesis", "")
                label_idx = row.get("label", -1)
                if label_idx not in (0, 1, 2):
                    continue
                state = f"Premise: {premise}\nHypothesis: {hypothesis}"
                yield TypedQuestion.noul(
                    id=f"anli-r{round_num}-{split_prefix}-{i:05d}",
                    state=state,
                    instructions="Does the premise entail the hypothesis?",
                    label=(label_idx == 0),
                    source="anli",
                    split=out_split,
                    group=f"anli-r{round_num}",
                )


# --- BoolQ (Boolean Questions) ---


def load_boolq() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("google/boolq", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            passage = row.get("passage", "")
            question = row.get("question", "")
            answer = row.get("answer", None)
            if answer is None or not passage or not question:
                continue
            state = f"Passage: {passage}\n\nQuestion: {question}"
            yield TypedQuestion.noul(
                id=f"boolq-{split_name}-{i:05d}",
                state=state,
                instructions="Based on the passage, is the answer yes?",
                label=bool(answer),
                source="boolq",
                split=out_split,
                group=f"boolq-{i % 100}",
            )


# --- Unified loader ---

# --- QuALITY (Long Document QA, Choice) ---

_QUALITY_KEYS = ["A", "B", "C", "D"]


def load_quality() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("emozilla/quality", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            article = row.get("article", "")
            question = row.get("question", "")
            options = row.get("options", [])
            answer = row.get("gold_label", -1)
            if not article or not question or len(options) != 4:
                continue
            if isinstance(answer, int) and answer in range(1, 5):
                answer_idx = answer - 1
            else:
                continue
            criteria = {_QUALITY_KEYS[j]: options[j] for j in range(4)}
            state = f"{article}\n\nQuestion: {question}"
            yield TypedQuestion.choice(
                id=f"quality-{split_name}-{i:05d}",
                state=state,
                instructions="Which answer is correct based on the passage?",
                criteria=criteria,
                label=_QUALITY_KEYS[answer_idx],
                source="quality",
                split=out_split,
                group=f"quality-{row.get('set_unique_id', i)}",
            )


# --- HotpotQA (Multi-hop Reasoning, Noul) ---


def load_hotpotqa() -> Iterator[TypedQuestion]:
    rng = random.Random(42)
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("hotpotqa/hotpot_qa", "distractor", split_name)
        out_split = "test" if split_name == "validation" else "train"
        if split_name == "train" and len(rows) > 10_000:
            rows = rng.sample(rows, 10_000)
        for i, row in enumerate(rows):
            question = row.get("question", "")
            answer = row.get("answer", "")
            context_titles = row.get("context", {}).get("title", [])
            context_sents = row.get("context", {}).get("sentences", [])
            if not question or not answer or not context_sents:
                continue
            paragraphs = []
            for title, sents in zip(context_titles, context_sents):
                paragraphs.append(f"{title}: {''.join(sents)}")
            state = "\n\n".join(paragraphs) + f"\n\nQuestion: {question}"
            is_yes_no = answer.lower() in ("yes", "no")
            if is_yes_no:
                yield TypedQuestion.noul(
                    id=f"hotpotqa-{split_name}-{i:05d}",
                    state=state,
                    instructions=f"Based on the passages, is the answer '{answer.lower()}'?",
                    label=(answer.lower() == "yes"),
                    source="hotpotqa",
                    split=out_split,
                    group=f"hotpotqa-{row.get('type', 'bridge')}",
                )
            else:
                yield TypedQuestion.noul(
                    id=f"hotpotqa-{split_name}-{i:05d}",
                    state=state,
                    instructions=f"Based on the passages, is '{answer}' the correct answer?",
                    label=True,
                    source="hotpotqa",
                    split=out_split,
                    group=f"hotpotqa-{row.get('type', 'bridge')}",
                )


# --- DROP (Discrete Reasoning Over Paragraphs, Noul/Score) ---


def load_drop() -> Iterator[TypedQuestion]:
    for split_name in ("train", "validation"):
        rows = _load_hf_parquet("ucinlp/drop", "default", split_name)
        out_split = "test" if split_name == "validation" else "train"
        for i, row in enumerate(rows):
            passage = row.get("passage", "")
            question = row.get("question", "")
            answers = row.get("answers_spans", {})
            spans = answers.get("spans", [])
            if not passage or not question or not spans:
                continue
            answer = spans[0]
            state = f"{passage}\n\nQuestion: {question}"
            try:
                num = float(answer.replace(",", ""))
                yield TypedQuestion.score(
                    id=f"drop-{split_name}-{i:05d}",
                    state=state,
                    instructions=f"What is the numerical answer? (expected: {num})",
                    criteria=[
                        "Incorrect: answer is wrong",
                        "Close: answer is approximately correct",
                        "Exact: answer matches exactly",
                    ],
                    label=2.0,
                    source="drop",
                    split=out_split,
                    group=f"drop-{i % 100}",
                )
            except ValueError:
                yield TypedQuestion.noul(
                    id=f"drop-{split_name}-{i:05d}",
                    state=state,
                    instructions=f"Based on the passage, is '{answer}' the correct answer?",
                    label=True,
                    source="drop",
                    split=out_split,
                    group=f"drop-{i % 100}",
                )


# --- Synthetic (generated data) ---


def load_synthetic() -> Iterator[TypedQuestion]:
    synth_path = DATA_DIR / "synthetic" / "synthetic.jsonl"
    if not synth_path.exists():
        return
    with synth_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield TypedQuestion(**d)


LOADERS = {
    "banking77": load_banking77,
    "sst2": load_sst2,
    "agnews": load_agnews,
    "mnli": load_mnli,
    "typed_decisions": load_typed_decisions,
    "stsb": load_stsb,
    "sst5": load_sst5,
    "arc": load_arc,
    "race": load_race,
    "hellaswag": load_hellaswag,
    "tabfact": load_tabfact,
    "fever": load_fever,
    "yelp": load_yelp,
    "swag": load_swag,
    "multirc": load_multirc,
    "mednli": load_mednli,
    "contractnli": load_contractnli,
    "codesearchnet": load_codesearchnet,
    "mmlu": load_mmlu,
    "winogrande": load_winogrande,
    "piqa": load_piqa,
    "commonsenseqa": load_commonsenseqa,
    "logiqa": load_logiqa,
    "anli": load_anli,
    "boolq": load_boolq,
    "quality": load_quality,
    "hotpotqa": load_hotpotqa,
    "drop": load_drop,
    "synthetic": load_synthetic,
}


def load_all(sources: list[str] | None = None) -> list[TypedQuestion]:
    selected = sources or list(LOADERS.keys())
    results = []
    for source in selected:
        loader = LOADERS.get(source)
        if loader is None:
            raise ValueError(f"Unknown source: {source}. Available: {list(LOADERS.keys())}")
        results.extend(loader())
    return results


def save_jsonl(items: list[TypedQuestion], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    CHUNK = 8192
    with path.open("w", encoding="utf-8", buffering=CHUNK * 1024) as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False))
            f.write("\n")
    return len(items)


def load_jsonl(path: Path) -> list[TypedQuestion]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(TypedQuestion(**d))
    return items


if __name__ == "__main__":
    sources = sys.argv[1:] or None
    items = load_all(sources)
    by_source = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    for source, source_items in sorted(by_source.items()):
        by_split = {}
        for item in source_items:
            by_split.setdefault(item.split, []).append(item)
        for split, split_items in sorted(by_split.items()):
            types = {}
            for item in split_items:
                types[item.question["type"]] = types.get(item.question["type"], 0) + 1
            print(f"  {source}/{split}: {len(split_items)} items ({types})")
        out = DATA_DIR / f"{source}.jsonl"
        n = save_jsonl(source_items, out)
        print(f"  → saved {n} items to {out}")
