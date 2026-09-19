"""Route B logit scorer: score options by reading log-probabilities directly.

Given a state and a question with typed options, this scorer:
1. Encodes the state + question as a shared prefix
2. For each option, computes the mean log-probability of its tokens
   conditioned on the prefix
3. Applies softmax normalization across options

No training required — uses the pretrained LM's own token predictions.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase


class LogitScorer:
    """Score typed-question options using direct LM log-probabilities.

    Two scoring strategies for choice questions:
      - "label": list options as "0. description", score label tokens ("0", "1", ...)
        Works for high-cardinality sets where labels are single tokens.
      - "description": score the full description text as a continuation
        More semantic signal per token, matches OpenJev's approach.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | None = None,
        norm: str = "mean",
        strategy: str = "description",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.norm = norm
        self.strategy = strategy

    def _score_options(self, context: str, options: list[str]) -> list[float]:
        """Compute normalized log-probability scores for each option.

        Uses DynamicCache.crop() to reuse the prefix KV cache across options
        without deep-copying tensors.
        """
        ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)

        cache = DynamicCache()
        with torch.no_grad():
            ctx_out = self.model(ctx_ids, past_key_values=cache, use_cache=True)
            last_logit = ctx_out.logits[0, -1]

        prefix_len = cache.get_seq_length()

        scores = []
        for opt_text in options:
            opt_ids = self.tokenizer(
                opt_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(self.device)

            if len(opt_ids) == 0:
                scores.append(float("-inf"))
                continue

            suffix_len = len(opt_ids)
            attn_mask = torch.ones((1, prefix_len + suffix_len), dtype=torch.long, device=self.device)
            cache_pos = torch.arange(prefix_len, prefix_len + suffix_len, device=self.device)

            with torch.no_grad():
                opt_out = self.model(
                    opt_ids.unsqueeze(0),
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )

            tokens_added = cache.get_seq_length() - prefix_len
            if tokens_added > 0:
                cache.crop(-tokens_added)

            logits = torch.cat([last_logit.unsqueeze(0), opt_out.logits[0, :-1]], dim=0)
            logp = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = logp[torch.arange(len(opt_ids), device=self.device), opt_ids]

            if self.norm == "mean":
                scores.append(tok_lp.mean().item())
            elif self.norm == "sum":
                scores.append(tok_lp.sum().item())
            else:
                scores.append(tok_lp.mean().item())

        return scores

    def _softmax(self, scores: list[float]) -> list[float]:
        t = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(t, dim=0).tolist()
        return probs

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Evaluate typed questions against a state.

        This implements the DecisionBackend protocol from model.src.server.
        """
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

    def _eval_noul(
        self, state: Any, instructions: str, criteria: dict[str, str] | None = None
    ) -> dict[str, Any]:
        true_desc = "yes"
        false_desc = "no"
        if criteria:
            true_desc = criteria.get("true", "yes")
            false_desc = criteria.get("false", "no")

        context = f"State: {state}\nQuestion: {instructions}\nAnswer:"
        scores = self._score_options(context, [f" {true_desc}", f" {false_desc}"])
        probs = self._softmax(scores)
        noul = round(max(0.01, min(0.99, probs[0])), 2)
        return {"type": "noul", "noul": noul}

    def _eval_choice(
        self, state: Any, instructions: str, criteria: dict[str, str]
    ) -> dict[str, Any]:
        keys = list(criteria.keys())
        descriptions = list(criteria.values())

        if self.strategy == "description":
            context = f"State: {state}\nQuestion: {instructions}\nAnswer:"
            option_texts = [f" {desc or key}" for key, desc in zip(keys, descriptions)]
        else:
            if len(keys) <= 10:
                labels = [str(i) for i in range(len(keys))]
            else:
                labels = [chr(65 + i) if i < 26 else f"opt{i}" for i in range(len(keys))]
            context = f"State: {state}\nQuestion: {instructions}\nOptions:\n"
            context += "\n".join(
                f"{labels[i]}. {descriptions[i] or keys[i]}" for i in range(len(keys))
            )
            context += "\nAnswer:"
            option_texts = [f" {label}" for label in labels]

        scores = self._score_options(context, option_texts)
        probs = self._softmax(scores)

        prob_dict = {k: round(p, 2) for k, p in zip(keys, probs)}
        choice = max(prob_dict, key=prob_dict.get)
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": prob_dict,
        }

    def _eval_score(
        self, state: Any, instructions: str, criteria: list[str]
    ) -> dict[str, Any]:
        context = f"State: {state}\nQuestion: {instructions}\nLevels:\n"
        context += "\n".join(f"{i}. {desc}" for i, desc in enumerate(criteria))
        context += "\nLevel:"

        option_texts = [f" {i}" for i in range(len(criteria))]
        scores = self._score_options(context, option_texts)
        probs = self._softmax(scores)

        prob_dict = {str(i): round(p, 2) for i, p in enumerate(probs)}
        legend = {str(i): desc for i, desc in enumerate(criteria)}
        score_val = sum(i * p for i, p in enumerate(probs))
        return {
            "type": "score",
            "score": round(score_val, 2),
            "probabilities": prob_dict,
            "legend": legend,
        }

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        """Single-question interface for evaluation suite compatibility."""
        result = self.evaluate(state, {"q": question})
        return result["q"]
