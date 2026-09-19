"""Encoder-based scorer: score options via bidirectional representation similarity.

Unlike LogitScorer (which reads continuation log-probabilities from a causal LM),
this scorer uses a bidirectional encoder to compare state+question against each option.

Two modes:
  - "biencoder": encode state+question and each option separately, score by cosine similarity
  - "crossencoder": encode [state+question + option] as one sequence per option, score by [CLS] projection

No training required for either mode — uses pretrained representations directly.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_encoder(
    model_name: str = "answerdotai/ModernBERT-base",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


class EncoderScorer:
    """Score typed-question options using bidirectional encoder representations."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | None = None,
        mode: str = "crossencoder",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.mode = mode

    def _mean_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    def _encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        return self._mean_pool(inputs["input_ids"], inputs["attention_mask"])[0]

    def _score_options_crossencoder(self, context: str, options: list[str]) -> list[float]:
        scores = []
        for opt in options:
            text = f"{context} {opt}"
            emb = self._encode(text)
            scores.append(emb.norm().item())
        return scores

    def _score_options_biencoder(self, context: str, options: list[str]) -> list[float]:
        query_emb = self._encode(context)
        query_emb = query_emb / query_emb.norm().clamp(min=1e-9)
        scores = []
        for opt in options:
            opt_emb = self._encode(opt)
            opt_emb = opt_emb / opt_emb.norm().clamp(min=1e-9)
            similarity = torch.dot(query_emb, opt_emb).item()
            scores.append(similarity)
        return scores

    def _score_options(self, context: str, options: list[str]) -> list[float]:
        if self.mode == "biencoder":
            return self._score_options_biencoder(context, options)
        return self._score_options_crossencoder(context, options)

    def _softmax(self, scores: list[float], temperature: float = 1.0) -> list[float]:
        t = torch.tensor(scores, dtype=torch.float32) / temperature
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
