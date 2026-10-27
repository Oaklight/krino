"""Low-rank adaptation (LoRA) for frozen backbones.

Implements LoRA directly on top of ``torch.nn.Linear`` rather than depending
on ``peft``, so the training hosts need no extra package.  An adapter wraps a
frozen linear layer and adds a trainable low-rank update:

    y = W x + (alpha / rank) * B A x

``B`` starts at zero, so an injected adapter is exactly the identity at
initialisation and training begins from the pretrained backbone's behaviour.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
arXiv:2106.09685.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Module-name segments that introduce a transformer block index, e.g.
# "model.layers.12.self_attn.q_proj" (Qwen, Llama) or
# "encoder.layer.5.attention.self.query" (BERT-style encoders).
_BLOCK_SEGMENTS = frozenset({"layers", "layer", "h", "blocks"})


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    Args:
        base: Linear layer to adapt.  Its parameters are frozen in place.
        rank: Bottleneck dimension of the update.
        alpha: Scaling numerator; the update is multiplied by ``alpha / rank``.
        dropout: Dropout applied to the input before the down-projection.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # The factors stay in fp32 so AdamW keeps precise moments even when
        # the backbone runs in bf16; the delta is cast back before the add.
        device = base.weight.device
        self.lora_a = nn.Linear(
            base.in_features, rank, bias=False, device=device, dtype=torch.float32
        )
        self.lora_b = nn.Linear(
            rank, base.out_features, bias=False, device=device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``base(x)`` plus the scaled low-rank update."""
        out = self.base(x)
        delta = self.lora_b(self.lora_a(self.dropout(x).float())) * self.scaling
        return out + delta.to(out.dtype)


def _layer_index(name: str) -> int | None:
    """Return the transformer block index encoded in a module name, if any."""
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in _BLOCK_SEGMENTS and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule registered at ``name`` with ``module``."""
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_lora(
    model: nn.Module,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    last_n_layers: int | None = None,
) -> int:
    """Replace matching linear layers with :class:`LoRALinear` adapters.

    Args:
        model: Backbone module, modified in place.
        targets: Module-name suffixes to adapt, e.g. ``("q_proj", "v_proj")``.
        rank: Bottleneck dimension of the update.
        alpha: Scaling numerator.
        dropout: Dropout on the adapter input.
        last_n_layers: Adapt only the last ``N`` transformer blocks.  ``None``
            adapts every matching layer in the model.

    Returns:
        Number of layers adapted.

    Raises:
        ValueError: If nothing matches ``targets``, or if ``last_n_layers`` is
            set but no matched module name carries a block index.  Either case
            would otherwise train nothing while appearing to work.
    """
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    if not matches:
        raise ValueError(f"No linear layers match LoRA targets {targets}")

    if last_n_layers is not None:
        indices = [i for i in (_layer_index(name) for name, _ in matches) if i is not None]
        if not indices:
            raise ValueError(
                "last_n_layers was set but no matched module name carries a block index"
            )
        cutoff = max(indices) - last_n_layers + 1
        matches = [
            (name, module)
            for name, module in matches
            if (idx := _layer_index(name)) is not None and idx >= cutoff
        ]

    for name, module in matches:
        _set_submodule(model, name, LoRALinear(module, rank, alpha, dropout))
    return len(matches)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the trainable LoRA parameters of ``model``."""
    return [
        param
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    ]
