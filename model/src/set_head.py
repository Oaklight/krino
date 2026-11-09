"""Set-function head over frozen option states (#340, arm D).

The input is the K option states of one panel-2 prompt, in natural option
order.  The head is a permutation-equivariant function of that set, following
the shape #340 pre-registers: a projection to a narrow width, a small set
transformer whose self-attention lets the options interact, one learned query
that pools the set into a single vector, and a dot-product readout of each
option against that vector.

Nothing in it uses an option's index.  Permuting the input permutes the K
logits and leaves the pooled vector unchanged, so order sensitivity has exactly
one source in this arm: the frozen tower's position encoding, which makes an
option's state differ when it is shown in a different place.  That is the
quantity the reversal measurement reports.

Config is fixed and pre-registered in #340: width 192, 2 layers, 4 heads,
feed-forward width 384, dropout 0, pooling ``pma_learned_query``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_WIDTH = 192
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 4
DEFAULT_FFN = 384


class SetBlock(nn.Module):
    """Pre-LN self-attention over the option set, then a feed-forward."""

    def __init__(self, width: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width)
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``values`` is ``(B, K, W)``; ``valid`` is ``(B, K)`` bool."""
        normalised = self.norm_attention(values)
        attended, _ = self.attention(
            normalised,
            normalised,
            normalised,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + attended
        return values + self.ffn(self.norm_ffn(values))


class SetHead(nn.Module):
    """K option states in, K logits out, equivariant to option order."""

    def __init__(
        self,
        hidden_size: int,
        width: int = DEFAULT_WIDTH,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        ffn: int = DEFAULT_FFN,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, width)
        self.blocks = nn.ModuleList(
            [SetBlock(width, heads, ffn) for _ in range(layers)]
        )
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.pool = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm_options = nn.LayerNorm(width)
        self.norm_pooled = nn.LayerNorm(width)

    def pooled(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """The permutation-invariant set summary, ``(B, W)``."""
        query = self.query.expand(values.shape[0], -1, -1)
        summary, _ = self.pool(
            query, values, values, key_padding_mask=~valid, need_weights=False
        )
        return summary[:, 0]

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Logits over the K options, with invalid slots at ``-inf``.

        Args:
            states: ``(B, K, H)`` float option states in natural order.
            valid: ``(B, K)`` bool, true for real options.

        Returns:
            ``(B, K)`` logits; masked slots are ``-inf`` so a softmax over the
            last dimension ignores them.
        """
        values = self.projection(states.float())
        for block in self.blocks:
            values = block(values, valid)
        summary = self.pooled(values, valid)
        # Dot product between each option and the same pooled vector: permuting
        # the options permutes the logits and leaves the summary unchanged.
        logits = (
            self.norm_options(values) * self.norm_pooled(summary).unsqueeze(1)
        ).sum(-1) / math.sqrt(values.shape[-1])
        return logits.masked_fill(~valid, float("-inf"))

    def probabilities(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Softmax over the valid options."""
        return torch.softmax(self.forward(states, valid), dim=-1)
