"""Per-type accuracy metrics for typed decision models."""

from __future__ import annotations

from typing import Any


def noul_accuracy(predictions: list[float], labels: list[bool], threshold: float = 0.5) -> dict[str, Any]:
    if not predictions:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for p, l in zip(predictions, labels) if (p > threshold) == l)
    return {"n": len(predictions), "accuracy": correct / len(predictions), "threshold": threshold}


def choice_accuracy(predictions: list[str], labels: list[str]) -> dict[str, Any]:
    if not predictions:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return {"n": len(predictions), "accuracy": correct / len(predictions)}


def score_mae(predictions: list[float], labels: list[float]) -> dict[str, Any]:
    if not predictions:
        return {"n": 0, "mae": 0.0}
    errors = [abs(p - l) for p, l in zip(predictions, labels)]
    return {"n": len(predictions), "mae": sum(errors) / len(errors)}


def evaluate_typed(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    noul_preds, noul_labels = [], []
    choice_preds, choice_labels = [], []
    score_preds, score_labels = [], []

    for pred, label in zip(predictions, labels):
        q_type = label.get("type", pred.get("type", ""))
        if q_type == "noul":
            noul_preds.append(pred.get("noul", 0.5))
            noul_labels.append(label.get("label", False))
        elif q_type == "choice":
            choice_preds.append(pred.get("choice", ""))
            choice_labels.append(label.get("label", ""))
        elif q_type == "score":
            score_preds.append(pred.get("score", 0.0))
            score_labels.append(label.get("label", 0.0))

    return {
        "noul": noul_accuracy(noul_preds, noul_labels) if noul_preds else None,
        "choice": choice_accuracy(choice_preds, choice_labels) if choice_preds else None,
        "score": score_mae(score_preds, score_labels) if score_preds else None,
        "total": len(predictions),
    }
