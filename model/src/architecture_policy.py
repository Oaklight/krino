"""Which declared architectures the loader may load directly.

``AutoModelForCausalLM`` covers every text decoder we train on, including Qwen3.5,
whose config also declares a ``*ForConditionalGeneration`` class. Loading by
architecture must therefore be an allowlist, not a suffix rule: a suffix rule
would move the incumbent off the causal-LM class and pull in a vision tower that
no text path asks for.

This module imports nothing from torch so the policy is testable without the
model stack installed.
"""

from __future__ import annotations

from typing import Any, Sequence

# Multimodal checkpoints that accept text-only inputs and return ``.logits``.
# Each entry is here because that property was checked on the checkpoint, not
# because the class name looks similar.
TEXT_DECODER_CONDITIONAL_ARCHITECTURES = frozenset({"Mistral3ForConditionalGeneration"})


def declared_architecture(config: Any) -> str | None:
    """Return the first declared architecture name, or ``None``."""
    architectures: Sequence[str] | None = getattr(config, "architectures", None)
    if not architectures:
        return None
    return architectures[0]


def routes_to_conditional_generation(config: Any) -> bool:
    """Whether this config must be loaded through its declared class.

    True only for an allowlisted architecture.  ``Qwen3_5ForConditionalGeneration``
    is deliberately not on the list even though the suffix matches.
    """
    return declared_architecture(config) in TEXT_DECODER_CONDITIONAL_ARCHITECTURES
