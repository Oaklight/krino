"""Shared construction of a :class:`DecisionModel` from resolved settings.

Training and evaluation must build the same architecture, or a checkpoint
loads into a graph with differently shaped heads and ``load_heads`` either
fails noisily or, worse, silently leaves some parameters at their initial
values.  Keeping the backbone selection, LoRA setup and head wiring in one
function makes that impossible to fork by accident.
"""

from __future__ import annotations

from .backbone import load_causal_lm, load_encoder, load_qwen35_base
from .decision_model import DecisionModel
from .gpu_config import GPUConfig


def build_decision_model(
    *,
    model_name: str,
    gpu_config: GPUConfig,
    device: str | None = None,
    use_encoder: bool = False,
    qwen35_base: bool = False,
    rank: int = 64,
    rival_aware: bool = False,
    mlp_layers: int = 0,
    mlp_dim: int | None = None,
    noul_rank: int | None = None,
    max_length: int | None = None,
    shared_attention: bool = False,
    unified_head: bool = False,
    readout: str = "last",
    readout_n: int = 4,
    lora_layers: int | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
    choice_mode: str = "tower",
) -> DecisionModel:
    """Load a frozen backbone and attach the typed decision heads.

    Args:
        model_name: HuggingFace backbone identifier.
        gpu_config: Resolved GPU settings (flash attention, bf16, max length).
        device: Torch device string. ``None`` selects CUDA when available.
        use_encoder: Load a bidirectional encoder instead of a causal LM.
        qwen35_base: Load a Qwen3.5 hybrid backbone.
        rank: Attention-head rank.
        rival_aware: Let options attend to each other before scoring.
        mlp_layers: MLP projector layers between backbone and heads.
        mlp_dim: Hidden width of the projector.
        noul_rank: Hidden width of the noul MLP; ``None`` uses a linear head.
        max_length: Requested sequence length; the model clamps it.
        shared_attention: Share one AttentionHead between choice and score.
        unified_head: Use one head for all three types.
        readout: Layer-mixing mode (``last``, ``mean_last_n``, ``weighted``).
        readout_n: Layer window for ``mean_last_n``.
        lora_layers: ``-1`` adapts every block, ``N`` the last N, ``None`` off.
        lora_rank: LoRA bottleneck rank.
        lora_alpha: LoRA scaling numerator.
        lora_dropout: Dropout on the LoRA input.
        lora_targets: Linear-layer suffixes that receive adapters.
        choice_mode: ``"tower"`` (state and options encoded separately, as
            before) or ``"cross_encoder"`` (one prompt per option with the state
            inside it; see :class:`model.src.heads.CrossEncoderHead`).

    Returns:
        The constructed model, still on the backbone's device.
    """
    flash_attention = gpu_config.flash_attention
    if qwen35_base:
        backbone, tokenizer = load_qwen35_base(
            model_name,
            device=device,
            freeze=True,
            flash_attention=flash_attention,
        )
    elif use_encoder:
        backbone, tokenizer = load_encoder(
            model_name,
            device=device,
            freeze=True,
            flash_attention=flash_attention,
        )
    else:
        backbone, tokenizer = load_causal_lm(
            model_name,
            device=device,
            freeze=True,
            flash_attention=flash_attention,
        )

    return DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=rank,
        rival_aware=rival_aware,
        mlp_layers=mlp_layers,
        mlp_dim=mlp_dim,
        noul_rank=noul_rank,
        max_length=max_length,
        gpu_config=gpu_config,
        shared_attention=shared_attention,
        unified_head=unified_head,
        readout=readout,
        readout_n=readout_n,
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        choice_mode=choice_mode,
    )
