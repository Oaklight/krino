"""Reranker-based scorer: score options via cross-encoder relevance scoring.

Uses sentence_transformers.CrossEncoder (e.g. Ettin rerankers) to score
(context, option) pairs. Each option is scored independently against the
context, and scores are softmaxed into probabilities.
"""

from __future__ import annotations

from typing import Any

import torch
from sentence_transformers import CrossEncoder

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_reranker(
    model_name: str = "cross-encoder/ettin-reranker-150m-v1",
    device: str | None = None,
    dtype: str = "float32",
) -> CrossEncoder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device, model_kwargs={"torch_dtype": _DTYPE_MAP[dtype]})


class RerankerScorer:
    """Score typed-question options using a cross-encoder reranker."""

    def __init__(self, model: CrossEncoder) -> None:
        self.model = model

    def _score_options(self, context: str, options: list[str]) -> list[float]:
        pairs = [(context, opt) for opt in options]
        scores = self.model.predict(pairs)
        return scores.tolist()

    def _softmax(self, scores: list[float]) -> list[float]:
        t = torch.tensor(scores, dtype=torch.float32)
        return torch.softmax(t, dim=0).tolist()

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        answers = {}
        for qid, question in questions.items():
            q_type = question.get("type", "noul")
            instructions = question.get("instructions", "")
            if q_type == "noul":
                answers[qid] = self._eval_noul(state, instructions, question.get("criteria"))
            elif q_type == "choice":
                answers[qid] = self._eval_choice(state, instructions, question["criteria"])
            elif q_type == "score":
                answers[qid] = self._eval_score(state, instructions, question["criteria"])
        return answers

    def _eval_noul(self, state: Any, instructions: str, criteria: dict | None = None) -> dict[str, Any]:
        true_desc = criteria.get("true", "yes") if criteria else "yes"
        false_desc = criteria.get("false", "no") if criteria else "no"
        context = f"{state} {instructions}"
        scores = self._score_options(context, [true_desc, false_desc])
        probs = self._softmax(scores)
        noul = round(max(0.01, min(0.99, probs[0])), 2)
        return {"type": "noul", "noul": noul}

    def _eval_choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
        keys = list(criteria.keys())
        descriptions = list(criteria.values())
        context = f"{state} {instructions}"
        option_texts = [desc or key for key, desc in zip(keys, descriptions)]
        scores = self._score_options(context, option_texts)
        probs = self._softmax(scores)
        prob_dict = {k: round(p, 2) for k, p in zip(keys, probs)}
        choice = max(prob_dict, key=prob_dict.get)
        return {"type": "choice", "choice": choice, "probabilities": prob_dict}

    def _eval_score(self, state: Any, instructions: str, criteria: list[str]) -> dict[str, Any]:
        context = f"{state} {instructions}"
        scores = self._score_options(context, criteria)
        probs = self._softmax(scores)
        prob_dict = {str(i): round(p, 2) for i, p in enumerate(probs)}
        legend = {str(i): desc for i, desc in enumerate(criteria)}
        score_val = sum(i * p for i, p in enumerate(probs))
        return {"type": "score", "score": round(score_val, 2), "probabilities": prob_dict, "legend": legend}

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        result = self.evaluate(state, {"q": question})
        return result["q"]
