"""Route B logit scorer: score options by reading log-probabilities directly.

Given a state and a question with typed options, this scorer:
1. Encodes the state + question as a shared prefix
2. For each option, computes the mean log-probability of its tokens
   conditioned on the prefix
3. Applies softmax normalization across options

No training required — uses the pretrained LM's own token predictions.
"""

from __future__ import annotations

import copy
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

    def _score_options(self, context: str, options: list[str], batch_size: int = 64) -> list[float]:
        """Compute normalized log-probability scores for each option.

        Batches options into a single forward pass per sub-batch to minimize
        GPU kernel launch overhead (77 sequential passes → 1-2 batched passes).
        """
        ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)

        cache = DynamicCache()
        with torch.no_grad():
            ctx_out = self.model(ctx_ids, past_key_values=cache, use_cache=True)
            last_logit = ctx_out.logits[0, -1]

        prefix_len = cache.get_seq_length()

        all_opt_ids = []
        for opt_text in options:
            ids = self.tokenizer(
                opt_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(self.device)
            all_opt_ids.append(ids)

        scores = [float("-inf")] * len(options)
        valid_indices = [i for i, ids in enumerate(all_opt_ids) if len(ids) > 0]

        for batch_start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[batch_start : batch_start + batch_size]
            batch_ids = [all_opt_ids[i] for i in batch_indices]
            is_last = batch_start + batch_size >= len(valid_indices)
            batch_scores = self._score_batch(cache, prefix_len, last_logit, batch_ids, copy_cache=not is_last)
            for i, idx in enumerate(batch_indices):
                scores[idx] = batch_scores[i]

        return scores

    def _score_batch(
        self,
        prefix_cache: DynamicCache,
        prefix_len: int,
        last_logit: torch.Tensor,
        opt_ids_list: list[torch.Tensor],
        copy_cache: bool = True,
    ) -> list[float]:
        n = len(opt_ids_list)
        lengths = [len(ids) for ids in opt_ids_list]
        max_len = max(lengths)

        pad_id = self.tokenizer.pad_token_id or 0
        padded = torch.full((n, max_len), pad_id, dtype=torch.long, device=self.device)
        for i, ids in enumerate(opt_ids_list):
            padded[i, : len(ids)] = ids

        # Attention mask: 1 for prefix + real tokens, 0 for padding
        attn_mask = torch.zeros((n, prefix_len + max_len), dtype=torch.long, device=self.device)
        for i, length in enumerate(lengths):
            attn_mask[i, : prefix_len + length] = 1

        cache_pos = torch.arange(prefix_len, prefix_len + max_len, device=self.device)

        batch_cache = copy.deepcopy(prefix_cache) if copy_cache else prefix_cache
        batch_cache.batch_repeat_interleave(n)

        with torch.no_grad():
            out = self.model(
                padded,
                attention_mask=attn_mask,
                past_key_values=batch_cache,
                cache_position=cache_pos,
                use_cache=False,
            )

        # Shifted logits: last_logit predicts token 0, out.logits[:, t-1] predicts token t
        # Build [n, max_len, vocab] prediction logits
        pred_logits = torch.cat(
            [last_logit.unsqueeze(0).unsqueeze(0).expand(n, 1, -1), out.logits[:, :-1, :]],
            dim=1,
        )
        logp = torch.log_softmax(pred_logits.float(), dim=-1)

        # Gather log-probs at actual token positions
        tok_logp = logp.gather(2, padded.unsqueeze(-1)).squeeze(-1)

        # Mask out padding
        length_mask = torch.zeros((n, max_len), device=self.device)
        for i, length in enumerate(lengths):
            length_mask[i, :length] = 1.0

        tok_logp = tok_logp * length_mask

        scores = []
        for i, length in enumerate(lengths):
            lp = tok_logp[i, :length]
            if self.norm == "sum":
                scores.append(lp.sum().item())
            else:
                scores.append(lp.mean().item())

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
