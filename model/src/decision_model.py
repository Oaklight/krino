"""Decision model: frozen backbone + trainable typed heads.

Combines a pretrained LM backbone (frozen) with lightweight decision
heads (trainable) for typed-question scoring. Supports both causal
LM and encoder backbones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .backbone import is_hybrid_model
from .heads import ChoiceHead, NoulHead, ScoreHead


def _is_encoder_model(backbone: PreTrainedModel) -> bool:
    """Detect whether backbone is an encoder (bidirectional) or causal decoder."""
    return not getattr(backbone.config, "is_decoder", True)


class DecisionModel(nn.Module):
    """Frozen backbone + trainable decision heads.

    Supports both causal LM (Qwen, Llama, Gemma) and encoder (ModernBERT, BERT)
    backbones. Pooling strategy is detected automatically:
      - Causal: last-token pooling (last token sees full sequence)
      - Encoder: mean pooling (all tokens see all tokens)
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int | None = None,
        rank: int = 64,
        rival_aware: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size or backbone.config.hidden_size
        self.is_encoder = _is_encoder_model(backbone)
        self.is_hybrid = is_hybrid_model(backbone.config)

        self.noul_head = NoulHead(self.hidden_size, dropout)
        self.choice_head = ChoiceHead(self.hidden_size, rank, rival_aware, dropout)
        self.score_head = ScoreHead(self.hidden_size, rank, dropout)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        device = next(self.backbone.parameters()).device
        self.noul_head = self.noul_head.to(device).float()
        self.choice_head = self.choice_head.to(device).float()
        self.score_head = self.score_head.to(device).float()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _encode_text(
        self, text: str | list[str], max_length: int = 512
    ) -> torch.Tensor:
        """Encode text and return pooled hidden state per sequence.

        Causal LM: last-token pooling (last token attended to full sequence).
        Encoder: mean pooling (all tokens see all tokens bidirectionally).
        """
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.backbone(
                **inputs, output_hidden_states=True, use_cache=False
            )
        hidden = outputs.hidden_states[-1]
        if self.is_encoder:
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), seq_lengths
            ]
        return pooled.float()

    def _encode_with_sequence(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text and return full sequence hidden states [1, seq_len, hidden]."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.backbone(
                **inputs, output_hidden_states=True, use_cache=False
            )
        hidden = outputs.hidden_states[-1]
        return hidden.float()

    def forward_noul(self, state: str, instructions: str) -> torch.Tensor:
        """Returns logit [1, 1] for noul prediction."""
        text = f"{state} {instructions}"
        pooled = self._encode_text(text)
        return self.noul_head(pooled)

    def forward_choice(
        self, state: str, instructions: str, option_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_options] for choice prediction."""
        context_text = f"{state} {instructions}"
        context_hidden = self._encode_with_sequence(context_text)
        option_pooled = self._encode_text(option_texts)
        option_hidden = option_pooled.unsqueeze(0)
        return self.choice_head(context_hidden, option_hidden)

    def forward_score(
        self, state: str, instructions: str, level_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_levels] for score prediction."""
        context_text = f"{state} {instructions}"
        context_hidden = self._encode_with_sequence(context_text)
        level_pooled = self._encode_text(level_texts)
        level_hidden = level_pooled.unsqueeze(0)
        return self.score_head(context_hidden, level_hidden)

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        """Single-question inference for eval suite compatibility."""
        q_type = question.get("type", "noul")
        instructions = question.get("instructions", "")

        if q_type == "noul":
            logit = self.forward_noul(state, instructions)
            noul = torch.sigmoid(logit).item()
            return {"type": "noul", "noul": round(max(0.01, min(0.99, noul)), 2)}

        elif q_type == "choice":
            criteria = question["criteria"]
            keys = list(criteria.keys())
            option_texts = [v or k for k, v in criteria.items()]
            logits = self.forward_choice(state, instructions, option_texts)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            prob_dict = {k: round(p, 2) for k, p in zip(keys, probs)}
            return {
                "type": "choice",
                "choice": max(prob_dict, key=prob_dict.get),
                "probabilities": prob_dict,
            }

        elif q_type == "score":
            criteria = question["criteria"]
            logits = self.forward_score(state, instructions, criteria)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            prob_dict = {str(i): round(p, 2) for i, p in enumerate(probs)}
            legend = {str(i): desc for i, desc in enumerate(criteria)}
            score_val = sum(i * p for i, p in enumerate(probs))
            return {
                "type": "score",
                "score": round(score_val, 2),
                "probabilities": prob_dict,
                "legend": legend,
            }

        raise ValueError(f"Unknown question type: {q_type}")

    def save_heads(self, path: Path | str) -> None:
        """Save only head parameters to a checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        head_state = {k: v for k, v in self.state_dict().items() if "backbone" not in k}
        torch.save(head_state, path)

    def load_heads(self, path: Path | str) -> None:
        """Load head parameters from a checkpoint."""
        head_state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(head_state, strict=False)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
