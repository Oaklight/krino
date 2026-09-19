"""Decision heads for typed-question scoring.

Three heads, one per primitive:
  NoulHead: binary probability via sigmoid
  ChoiceHead: option selection with attention-based scoring
  ScoreHead: ordered level scoring (same mechanism as ChoiceHead)

All heads take hidden states from a frozen backbone and produce
probability distributions over the answer space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoulHead(nn.Module):
    """Binary probability head. Maps hidden state to P(yes) via sigmoid."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Args: hidden_state [batch, hidden_size]. Returns: logits [batch, 1]."""
        return self.proj(self.dropout(hidden_state))

    def predict(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Returns P(yes) in [0, 1]."""
        return torch.sigmoid(self.forward(hidden_state)).squeeze(-1)


class AttentionHead(nn.Module):
    """Attention-based option scorer adapted from jevlike.

    Each option embedding queries the context representation via
    cross-attention, producing one score per option. Options interact
    through the shared context representation but are scored in parallel.

    Optionally includes rival-aware attention (from jevbetter) where
    options attend to each other before scoring.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 64,
        rival_aware: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.rival_aware = rival_aware

        self.layer_norm_ctx = nn.LayerNorm(hidden_size)
        self.layer_norm_opt = nn.LayerNorm(hidden_size)

        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)

        if rival_aware:
            self.rival_query = nn.Linear(hidden_size, rank, bias=False)
            self.rival_key = nn.Linear(hidden_size, rank, bias=False)
            self.rival_value = nn.Linear(hidden_size, rank, bias=False)
            self.rival_gate = nn.Linear(rank * 2, rank)

        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5

    def forward(
        self,
        context_hidden: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score options against context.

        Args:
            context_hidden: [batch, seq_len, hidden_size] — backbone output for state+question
            option_hidden: [batch, n_options, hidden_size] — pooled option representations
            option_mask: [batch, n_options] — True for valid options

        Returns:
            logits: [batch, n_options] — one score per option (pre-softmax)
        """
        ctx = self.layer_norm_ctx(context_hidden)
        opt = self.layer_norm_opt(option_hidden)

        if self.rival_aware:
            opt = self._rival_attention(opt, option_mask)

        q = self.query(opt)
        k = self.key(ctx)
        v = self.value(ctx)

        attn_scores = torch.einsum("bnr,blr->bnl", q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.einsum("bnl,blr->bnr", attn_weights, v)

        logits = (q * attended).sum(-1)
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, float("-inf"))

        return logits

    def _rival_attention(self, opt: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        rq = self.rival_query(opt)
        rk = self.rival_key(opt)
        rv = self.rival_value(opt)

        rival_scores = torch.einsum("bnr,bmr->bnm", rq, rk) * self.scale
        if mask is not None:
            rival_scores = rival_scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        rival_weights = F.softmax(rival_scores, dim=-1)
        rival_ctx = torch.einsum("bnm,bmr->bnr", rival_weights, rv)

        q_orig = self.query(opt)
        gate = torch.sigmoid(self.rival_gate(torch.cat([q_orig, rival_ctx], dim=-1)))
        return opt + gate * rival_ctx


class ChoiceHead(nn.Module):
    """Choice head: scores K options via AttentionHead, returns softmax probabilities."""

    def __init__(self, hidden_size: int, rank: int = 64, rival_aware: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = AttentionHead(hidden_size, rank, rival_aware, dropout)

    def forward(
        self,
        context_hidden: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits [batch, n_options]."""
        return self.attention(context_hidden, option_hidden, option_mask)

    def predict(
        self,
        context_hidden: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns probabilities [batch, n_options]."""
        logits = self.forward(context_hidden, option_hidden, option_mask)
        return F.softmax(logits, dim=-1)


class ScoreHead(nn.Module):
    """Score head: same as ChoiceHead over ordered levels, plus expected-value computation."""

    def __init__(self, hidden_size: int, rank: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = AttentionHead(hidden_size, rank, rival_aware=False, dropout=dropout)

    def forward(
        self,
        context_hidden: torch.Tensor,
        level_hidden: torch.Tensor,
        level_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits [batch, n_levels]."""
        return self.attention(context_hidden, level_hidden, level_mask)

    def predict(
        self,
        context_hidden: torch.Tensor,
        level_hidden: torch.Tensor,
        level_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (probabilities [batch, n_levels], expected_score [batch])."""
        logits = self.forward(context_hidden, level_hidden, level_mask)
        probs = F.softmax(logits, dim=-1)
        n_levels = probs.shape[-1]
        level_indices = torch.arange(n_levels, device=probs.device, dtype=probs.dtype)
        expected = (probs * level_indices).sum(-1)
        return probs, expected
