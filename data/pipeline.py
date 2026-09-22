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


# --- Unified loader ---

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
