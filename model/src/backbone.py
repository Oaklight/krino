"""Model backbone loading and abstraction."""

from __future__ import annotations

import logging
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from model.src.architecture_policy import (
    declared_architecture,
    routes_to_conditional_generation,
)

try:  # transformers >= 4.45 exposes the image-text auto class
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - older transformers
    AutoModelForImageTextToText = None

try:  # Qwen3.5 4B ships as a multimodal conditional-generation model
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:  # pragma: no cover - older transformers
    try:
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError:
        Qwen3_5ForConditionalGeneration = None

logger = logging.getLogger(__name__)

#: Attributes that hold a nested text config on a composite model config.
_TEXT_CONFIG_ATTRS = ("text_config", "language_config", "llm_config")

#: Text fields ``DecisionModel`` and the hybrid check read off the backbone.
_TEXT_CONFIG_FIELDS = ("hidden_size", "num_hidden_layers", "layer_types", "vocab_size")

#: Architecture names that mean "this model is a causal decoder".
_CAUSAL_ARCH_SUFFIXES = ("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")


def _flash_attention_kwargs(flash_attention: bool) -> dict:
    """Build ``attn_implementation`` kwarg for efficient attention.

    Tries flash_attn package first (Flash Attention 2), falls back to
    PyTorch's built-in SDPA which includes Flash, math, and mem-efficient
    kernels without any extra installation.

    Args:
        flash_attention: Whether to enable efficient attention.

    Returns:
        Dict to merge into ``from_pretrained`` kwargs.
    """
    if not flash_attention:
        return {}
    try:
        import flash_attn as _  # noqa: F401

        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        return {"attn_implementation": "sdpa"}


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

    config = AutoConfig.from_pretrained(model_name)
    # Multimodal checkpoints that still expose a text decoder (Mistral3 today)
    # are not in AutoModelForCausalLM's mapping.  They accept text-only inputs
    # and return ``.logits``, so the declared architecture is loaded directly -
    # but only from an allowlist, never by suffix: Qwen3.5 declares
    # Qwen3_5ForConditionalGeneration too, and routing the incumbent through it
    # would pull a vision tower no text path asks for.
    loader = AutoModelForCausalLM
    if routes_to_conditional_generation(config):
        arch = declared_architecture(config)
        loader = getattr(transformers, arch, AutoModelForCausalLM)
    tok_kwargs = {}
    if config.model_type in ("mistral3", "pixtral"):
        # The released Mistral tokenizers carry a regex this transformers
        # version reads incorrectly; the flag selects the corrected pattern.
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    kwargs = {
        "dtype": dtype,
        "device_map": device if device == "auto" else None,
        **_flash_attention_kwargs(flash_attention),
    }
    model = loader.from_pretrained(model_name, **kwargs)
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


def _qwen35_loader_candidates() -> list[tuple[str, Any]]:
    """Auto classes that can load a Qwen3.5 checkpoint, in preference order.

    Text-only Qwen3.5 0.8B/2B checkpoints are ``*ForCausalLM`` and load with
    the first entry.  The 4B family is ``Qwen3_5ForConditionalGeneration``
    (multimodal), which the causal-LM auto class rejects, so the image-text
    auto class and the concrete class are the fallbacks.  Both are optional:
    older transformers releases expose neither.
    """
    candidates: list[tuple[str, Any]] = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    if AutoModelForImageTextToText is not None:
        candidates.append(
            ("AutoModelForImageTextToText", AutoModelForImageTextToText)
        )
    if Qwen3_5ForConditionalGeneration is not None:
        candidates.append(
            ("Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration)
        )
    return candidates


def _load_qwen35_full(
    model_name: str,
    *,
    dtype: torch.dtype,
    flash_attention: bool,
) -> Any:
    """Load a Qwen3.5 checkpoint with the first auto class that accepts it."""
    kwargs = {"dtype": dtype, **_flash_attention_kwargs(flash_attention)}
    errors: list[tuple[str, Exception]] = []
    for label, loader in _qwen35_loader_candidates():
        try:
            return loader.from_pretrained(model_name, **kwargs)
        except Exception as error:  # noqa: BLE001 - try the next architecture
            logger.debug("Qwen3.5 load via %s failed: %s", label, error)
            errors.append((label, error))
    if not errors:
        raise RuntimeError(
            f"No Qwen3.5 loader is available for {model_name!r}; install a "
            "transformers release that defines Qwen3_5ForConditionalGeneration"
        )
    detail = "; ".join(
        f"{label}: {type(error).__name__}: {error}" for label, error in errors
    )
    raise RuntimeError(
        f"Could not load {model_name!r} as any Qwen3.5 architecture ({detail})"
    ) from errors[-1][1]


def _extract_text_backbone(full_model: Any) -> Any:
    """Return the text tower inside a Qwen3.5 wrapper.

    Supported shapes, checked in order:

    - ``full_model.model.language_model`` (multimodal
      ``Qwen3_5ForConditionalGeneration``);
    - ``full_model.model.model.language_model`` (one extra nesting level);
    - ``full_model.model`` (text-only ``*ForCausalLM``);
    - ``full_model`` itself when it has no ``.model``.
    """
    outer = getattr(full_model, "model", None)
    holders = []
    if outer is not None:
        holders.append(outer)
        inner = getattr(outer, "model", None)
        if inner is not None and inner is not outer:
            holders.append(inner)
    holders.append(full_model)
    for holder in holders:
        language_model = getattr(holder, "language_model", None)
        if language_model is not None:
            return language_model
    return outer if outer is not None else full_model


def _extract_lm_head(full_model: Any, text_backbone: Any) -> Any | None:
    """Return the vocabulary head that belongs to the text tower, if any.

    The multimodal wrapper keeps ``lm_head`` on the top-level conditional
    generation module, while a causal-LM wrapper keeps it beside ``.model``.
    The text tower itself normally has none; returning ``None`` is valid.
    """
    for holder in (full_model, getattr(full_model, "model", None), text_backbone):
        if holder is None:
            continue
        lm_head = getattr(holder, "lm_head", None)
        if lm_head is not None:
            return lm_head
    return None


def _has_causal_architecture(names: Any) -> bool:
    """True when an ``architectures`` list names a causal decoder."""
    return any(
        str(name).endswith(_CAUSAL_ARCH_SUFFIXES) for name in (names or ())
    )


def _resolve_text_config(config: Any) -> Any:
    """Return the text config, not the multimodal config that wraps it.

    ``get_text_config()`` is the documented path but can raise when a config
    carries several text sections.  The explicit ``text_config`` /
    ``language_config`` / ``llm_config`` attributes are the fallback.  Fields
    that a nested text config omits (notably ``layer_types``) are copied from
    the parent, and the parent's causal ``architectures`` entry is copied when
    the nested config only names a bare ``*TextModel``; both are needed by
    ``DecisionModel`` to pick the causal branch.
    """
    if config is None:
        return None

    text_config = None
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            text_config = getter()
        except (TypeError, ValueError):
            text_config = None
    if text_config is None or text_config is config:
        for attr in _TEXT_CONFIG_ATTRS:
            nested = getattr(config, attr, None)
            if nested is not None and nested is not config:
                text_config = nested
                break
    if text_config is None:
        text_config = config

    for field in _TEXT_CONFIG_FIELDS:
        current = getattr(text_config, field, None)
        if current is None or (isinstance(current, (list, tuple)) and not current):
            value = getattr(config, field, None)
            if value is not None:
                try:
                    setattr(text_config, field, value)
                except (AttributeError, TypeError):
                    pass

    parent_architectures = getattr(config, "architectures", None)
    text_architectures = getattr(text_config, "architectures", None)
    if parent_architectures and not _has_causal_architecture(text_architectures):
        try:
            text_config.architectures = list(parent_architectures)
        except (AttributeError, TypeError):
            pass

    return text_config


def _load_qwen35_parts(
    model_name: str,
    device: str | None,
    dtype: torch.dtype,
    freeze: bool,
    flash_attention: bool,
    tokenizer: AutoTokenizer | None,
) -> tuple[Any, AutoTokenizer, Any | None]:
    """Load a Qwen3.5 checkpoint and split it into text tower, tokenizer, head."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_model = _load_qwen35_full(
        model_name, dtype=dtype, flash_attention=flash_attention
    )
    backbone = _extract_text_backbone(full_model)
    lm_head = _extract_lm_head(full_model, backbone)

    config = getattr(full_model, "config", None)
    if config is None:
        config = AutoConfig.from_pretrained(model_name)
    backbone.config = _resolve_text_config(config)

    if device != "auto":
        backbone = backbone.to(device)
        if lm_head is not None:
            lm_head = lm_head.to(device)
    backbone.eval()
    if lm_head is not None:
        lm_head.eval()

    if freeze:
        for param in backbone.parameters():
            param.requires_grad_(False)
        if lm_head is not None:
            for param in lm_head.parameters():
                param.requires_grad_(False)

    return backbone, tokenizer, lm_head


def load_qwen35_base(
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    flash_attention: bool = False,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load Qwen3.5 Base model (GDN hybrid architecture).

    Qwen3.5 uses a Gated DeltaNet + standard attention hybrid.  The loader is
    architecture-aware: it tries ``AutoModelForCausalLM`` first (text-only
    0.8B/2B), then ``AutoModelForImageTextToText`` and the concrete
    ``Qwen3_5ForConditionalGeneration`` (4B family, which wraps a vision tower
    around a text tower).  Whichever wrapper loads, the text tower is
    extracted and the text config is attached so that ``hidden_size`` and
    other attributes are accessible on ``backbone.config``.

    Args:
        model_name: HuggingFace model identifier (must be a -Base variant).
        device: Target device. None uses "cuda" if available, else "cpu".
        dtype: Model precision.
        freeze: If True, freeze all parameters (no gradients).
        flash_attention: If True, attempt to use Flash Attention 2.
        tokenizer: Optional pre-loaded tokenizer.  ``None`` loads one.

    Returns:
        (backbone, tokenizer) tuple where backbone is the text model.
    """
    backbone, tokenizer, _ = _load_qwen35_parts(
        model_name,
        device,
        dtype,
        freeze,
        flash_attention,
        tokenizer,
    )
    return backbone, tokenizer


def load_qwen35_base_with_head(
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    flash_attention: bool = False,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[AutoModel, AutoTokenizer, Any | None]:
    """Load a Qwen3.5 text tower together with its vocabulary head.

    This is the scoring variant of :func:`load_qwen35_base`: callers that read
    yes/no token logits (the raw reranker baseline) need the ``lm_head`` that
    lives on the wrapper, not on the extracted text tower.  The returned
    ``lm_head`` is ``None`` when the checkpoint exposes none.

    Returns:
        (backbone, tokenizer, lm_head) tuple.
    """
    return _load_qwen35_parts(
        model_name,
        device,
        dtype,
        freeze,
        flash_attention,
        tokenizer,
    )


def _looks_like_config(value: Any) -> bool:
    """True for a real config-like object, not a mock or a scalar.

    ``unittest.mock.MagicMock`` answers every attribute lookup with another
    mock, so an unguarded nested-config walk would recurse forever.
    """
    if value is None or isinstance(value, (str, bytes, int, float, bool, list, dict)):
        return False
    if type(value).__module__.startswith(("unittest.mock", "mock")):
        return False
    return any(
        hasattr(value, attr) for attr in ("hidden_size", "layer_types", "to_dict")
    )


def _iter_configs(config: Any):
    """Yield a config and every nested text config it carries."""
    seen: set[int] = set()
    stack = [config]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in _TEXT_CONFIG_ATTRS:
            nested = getattr(current, attr, None)
            if nested is not None and nested is not current and _looks_like_config(nested):
                stack.append(nested)


def is_hybrid_model(config: object) -> bool:
    """Check if model uses GDN hybrid attention (Qwen3.5+).

    GDN models mix ``"linear_attention"`` and standard ``"attention"``
    layers. The ``layer_types`` config attribute lists the type of each
    transformer block.  On a multimodal config the list lives on the nested
    text config, so every nested text section is checked.

    Args:
        config: A model config object (e.g. ``backbone.config``) or a
            composite config carrying a nested ``text_config``.

    Returns:
        True if the model has GDN linear-attention layers.
    """
    for candidate in _iter_configs(config):
        layer_types = getattr(candidate, "layer_types", None)
        if isinstance(layer_types, (list, tuple, set, frozenset)) and (
            "linear_attention" in layer_types
        ):
            return True
    return False
