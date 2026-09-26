"""Model backbone loading and abstraction."""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _flash_attention_kwargs(flash_attention: bool) -> dict:
    """Build ``attn_implementation`` kwarg if Flash Attention 2 is requested.

    Args:
        flash_attention: Whether to attempt Flash Attention 2.

    Returns:
        Dict to merge into ``from_pretrained`` kwargs.
    """
    if not flash_attention:
        return {}
    try:
        # Verify the flash_attn package is available before requesting it
        import flash_attn as _  # noqa: F401

        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        logger.warning(
            "flash_attention requested but flash_attn package not installed; "
            "falling back to default attention"
        )
        return {}


def load_causal_lm(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    flash_attention: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device. None uses "cuda" if available, else "cpu".
        dtype: Model precision.
        freeze: If True, freeze all parameters (no gradients).
        flash_attention: If True, attempt to use Flash Attention 2
            (requires ``flash_attn`` package and Ampere+ GPU).

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {
        "dtype": dtype,
        "device_map": device if device == "auto" else None,
        **_flash_attention_kwargs(flash_attention),
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
    flash_attention: bool = False,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load a bidirectional encoder model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device. None uses "cuda" if available, else "cpu".
        dtype: Model precision.
        freeze: If True, freeze all parameters (no gradients).
        flash_attention: If True, attempt to use Flash Attention 2.

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {
        "dtype": dtype,
        **_flash_attention_kwargs(flash_attention),
    }
    model = AutoModel.from_pretrained(model_name, **kwargs)
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
    flash_attention: bool = False,
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
        flash_attention: If True, attempt to use Flash Attention 2.

    Returns:
        (backbone, tokenizer) tuple where backbone is the text model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load as causal LM, then extract the text backbone
    # (.model strips the vocab/lm_head — we only need hidden states)
    kwargs = {
        "dtype": dtype,
        **_flash_attention_kwargs(flash_attention),
    }
    full_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
