"""Model backbone loading and abstraction."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
