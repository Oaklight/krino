"""Data pipeline: download, convert, and split public benchmarks into typed-question JSONL."""

from __future__ import annotations

import json
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
    print(f"  downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as resp:
        cache_path.write_bytes(resp.read())
    return cache_path


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for data preparation: pip install 'jev-explore[data]'")
    table = pq.read_table(path)
    return table.to_pylist()


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
        old_cache.rename(new_first)

    rows: list[dict[str, Any]] = []
    for shard_idx in range(1000):
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

        rows.extend(_read_parquet(cache_path))

    return rows


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


# --- Unified loader ---

LOADERS = {
    "banking77": load_banking77,
    "sst2": load_sst2,
    "agnews": load_agnews,
    "mnli": load_mnli,
    "typed_decisions": load_typed_decisions,
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
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
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
