"""Model backbone loading and abstraction."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device. None uses "cuda" if available, else "cpu".
        dtype: Model precision.
        freeze: If True, freeze all parameters (no gradients).

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)

    return model, tokenizer


def load_encoder(
    model_name: str = "answerdotai/ModernBERT-base",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load a bidirectional encoder model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    if device != "auto":
        model = model.to(device)
    model.eval()

    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)

    return model, tokenizer


def load_qwen35_base(
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load Qwen3.5 Base model (GDN hybrid architecture).

    Qwen3.5 uses a Gated DeltaNet + standard attention hybrid. We load
    via AutoModelForCausalLM then extract ``.model`` to get the text
    backbone without the LM head. The text config is attached so that
    ``hidden_size`` and other attributes are accessible on
    ``backbone.config``.

    Args:
        model_name: HuggingFace model identifier (must be a -Base variant).
        device: Target device. None uses "cuda" if available, else "cpu".
        dtype: Model precision.
        freeze: If True, freeze all parameters (no gradients).

    Returns:
        (backbone, tokenizer) tuple where backbone is the text model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load as causal LM, then extract the text backbone
    # (.model strips the vocab/lm_head — we only need hidden states)
    full_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    backbone = full_model.model

    # Attach text config so hidden_size etc. are accessible
    config = AutoConfig.from_pretrained(model_name)
    text_config = config.get_text_config()
    backbone.config = text_config

    if device != "auto":
        backbone = backbone.to(device)
    backbone.eval()

    if freeze:
        for param in backbone.parameters():
            param.requires_grad_(False)

    return backbone, tokenizer


def is_hybrid_model(config: object) -> bool:
    """Check if model uses GDN hybrid attention (Qwen3.5+).

    GDN models mix ``"linear_attention"`` and standard ``"attention"``
    layers. The ``layer_types`` config attribute lists the type of each
    transformer block.

    Args:
        config: A model config object (e.g. ``backbone.config``).

    Returns:
        True if the model has GDN linear-attention layers.
    """
    layer_types = getattr(config, "layer_types", None) or []
    return "linear_attention" in set(layer_types)
