"""Data pipeline: download, convert, and split public benchmarks into typed-question JSONL."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from .format import TypedQuestion

DATA_DIR = Path(__file__).resolve().parent / "benchmarks"


def _download_json(url: str, cache_path: Path) -> Any:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    cache_path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def _download_csv_lines(url: str, cache_path: Path) -> list[str]:
    if cache_path.exists():
        return cache_path.read_text().splitlines()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode()
    cache_path.write_text(text)
    return text.splitlines()


# --- Banking77 ---

BANKING77_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data"
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
        cache = DATA_DIR / "banking77" / f"{split_name}.json"
        url = f"{BANKING77_URL}/{split_name}-00000-of-00001.parquet"
        try:
            data = _download_json(
                f"https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/{split_name}",
                DATA_DIR / "banking77" / f"{split_name}_meta.json",
            )
            parquet_url = data[0]["url"] if data else url
        except Exception:
            parquet_url = url
        rows_cache = DATA_DIR / "banking77" / f"{split_name}_rows.json"
        if not rows_cache.exists():
            try:
                rows_data = _download_json(
                    f"https://datasets-server.huggingface.co/rows?dataset=PolyAI/banking77&config=default&split={split_name}&offset=0&length=10000",
                    rows_cache,
                )
            except Exception:
                rows_data = {"rows": []}
        else:
            rows_data = json.loads(rows_cache.read_text())
        for i, row_entry in enumerate(rows_data.get("rows", [])):
            row = row_entry.get("row", row_entry)
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
        rows_cache = DATA_DIR / "sst2" / f"{split_name}_rows.json"
        try:
            rows_data = _download_json(
                f"https://datasets-server.huggingface.co/rows?dataset=stanfordnlp/sst2&config=default&split={split_name}&offset=0&length=10000",
                rows_cache,
            )
        except Exception:
            rows_data = {"rows": []}
        out_split = "test" if split_name == "validation" else "train"
        for i, row_entry in enumerate(rows_data.get("rows", [])):
            row = row_entry.get("row", row_entry)
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
        rows_cache = DATA_DIR / "agnews" / f"{split_name}_rows.json"
        try:
            rows_data = _download_json(
                f"https://datasets-server.huggingface.co/rows?dataset=fancyzhx/ag_news&config=default&split={split_name}&offset=0&length=10000",
                rows_cache,
            )
        except Exception:
            rows_data = {"rows": []}
        for i, row_entry in enumerate(rows_data.get("rows", [])):
            row = row_entry.get("row", row_entry)
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
        rows_cache = DATA_DIR / "mnli" / f"{split_name}_rows.json"
        try:
            rows_data = _download_json(
                f"https://datasets-server.huggingface.co/rows?dataset=nyu-mll/multi_nli&config=default&split={split_name}&offset=0&length=10000",
                rows_cache,
            )
        except Exception:
            rows_data = {"rows": []}
        out_split = "test" if "validation" in split_name else "train"
        for i, row_entry in enumerate(rows_data.get("rows", [])):
            row = row_entry.get("row", row_entry)
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
        rows_cache = DATA_DIR / "typed_decisions" / f"{split_name}_rows.json"
        try:
            rows_data = _download_json(
                f"https://datasets-server.huggingface.co/rows?dataset=LocalLLaMA/typed-decisions&config=default&split={split_name}&offset=0&length=10000",
                rows_cache,
            )
        except Exception:
            rows_data = {"rows": []}
        for i, row_entry in enumerate(rows_data.get("rows", [])):
            row = row_entry.get("row", row_entry)
            state = row.get("state", row.get("context", ""))
            q = row.get("question", {})
            if isinstance(q, str):
                q = json.loads(q) if q.startswith("{") else {"type": "noul", "instructions": q}
            label = row.get("label", row.get("answer", None))
            yield TypedQuestion(
                id=f"typed_decisions-{split_name}-{i:05d}",
                state=state,
                question=q,
                label=label,
                source="typed_decisions",
                split=split_name,
                group=row.get("domain", f"td-{i % 20}"),
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
    import sys
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
