"""Head-to-head comparison against the Jev API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def compare_on_dataset(
    model_scorer: Any,
    dataset_path: Path,
    jev_api_key: str | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run the same typed-question items through both model and Jev API.

    Args:
        model_scorer: object with .predict(state, question) -> answer_dict
        dataset_path: path to typed-question JSONL
        jev_api_key: if provided, also run through Jev API for comparison
        max_items: limit items for quick tests
    """
    from data.pipeline import load_jsonl

    items = load_jsonl(dataset_path)
    if max_items:
        items = items[:max_items]

    model_results = []
    jev_results = []

    for item in items:
        model_answer = model_scorer.predict(item.state, item.question)
        model_results.append({"item_id": item.id, "answer": model_answer, "label": item.label, "type": item.question["type"]})

    if jev_api_key:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probing" / "scripts"))
        from jev_client import JevClient
        jev = JevClient(api_key=jev_api_key)
        for item in items:
            try:
                resp = jev.ask(item.state, {"q": item.question})
                jev_answer = resp["answers"]["q"]
                jev_results.append({"item_id": item.id, "answer": jev_answer, "label": item.label, "type": item.question["type"]})
            except Exception as e:
                jev_results.append({"item_id": item.id, "error": str(e), "type": item.question["type"]})
        jev.close()

    return {
        "items": len(items),
        "model_results": model_results,
        "jev_results": jev_results if jev_results else None,
    }
