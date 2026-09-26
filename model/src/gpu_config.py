"""GPU configuration and auto-detection for memory-safe training.

Detects GPU hardware (memory, compute capability) and returns optimal
settings for Flash Attention, bf16 compute, max sequence length, and
gradient checkpointing.  All optimizations are configurable via CLI
overrides; auto-detection provides sensible defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Optimal GPU settings for backbone forward passes.

    Attributes:
        flash_attention: Enable Flash Attention 2 (requires Ampere+).
        bf16_compute: Use bf16 autocast for backbone forward passes.
        max_length: Safe maximum sequence length for this GPU.
        gradient_checkpointing: Enable gradient checkpointing on backbone.
    """

    flash_attention: bool
    bf16_compute: bool
    max_length: int
    gradient_checkpointing: bool

    @classmethod
    def auto_detect(cls, model_params_billions: float = 0.6) -> GPUConfig:
        """Detect GPU hardware and return optimal config.

        Args:
            model_params_billions: Approximate backbone size in billions of
                parameters.  Used to estimate memory headroom after model
                loading (rough rule: ~7 bytes/param for bf16 model + optimizer
                states on frozen params is negligible, but activations scale
                with it).

        Returns:
            A GPUConfig with hardware-appropriate defaults.
        """
        if not torch.cuda.is_available():
            return cls(
                flash_attention=False,
                bf16_compute=False,
                max_length=512,
                gradient_checkpointing=False,
            )

        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = props.total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)

        # Flash Attention 2 requires compute capability >= 8.0 (Ampere+)
        flash_ok = compute_cap[0] >= 8
        # Native bf16 requires compute capability >= 8.0 (Ampere+)
        bf16_ok = compute_cap[0] >= 8

        # Estimate available memory after model loading.
        # Rough heuristic: bf16 model ~ 2 bytes/param, but with framework
        # overhead ~7 bytes/param is a conservative upper bound.
        available_gb = gpu_mem_gb - model_params_billions * 7

        if available_gb > 100:
            safe_max_length = 32768
        elif available_gb > 30:
            safe_max_length = 8192
        elif available_gb > 10:
            safe_max_length = 4096
        else:
            safe_max_length = 2048

        return cls(
            flash_attention=flash_ok,
            bf16_compute=bf16_ok,
            max_length=safe_max_length,
            gradient_checkpointing=available_gb < 20,
        )

    def apply_overrides(
        self,
        *,
        flash_attention: bool | None = None,
        bf16: bool | None = None,
        gradient_checkpointing: bool | None = None,
        max_length: int | None = None,
    ) -> GPUConfig:
        """Return a new GPUConfig with explicit CLI overrides applied.

        Only non-None arguments replace the auto-detected values.

        Args:
            flash_attention: Override flash attention setting.
            bf16: Override bf16 compute setting.
            gradient_checkpointing: Override gradient checkpointing setting.
            max_length: Override max sequence length.

        Returns:
            A new GPUConfig with overrides applied.
        """
        return GPUConfig(
            flash_attention=flash_attention if flash_attention is not None else self.flash_attention,
            bf16_compute=bf16 if bf16 is not None else self.bf16_compute,
            max_length=max_length if max_length is not None else self.max_length,
            gradient_checkpointing=(
                gradient_checkpointing
                if gradient_checkpointing is not None
                else self.gradient_checkpointing
            ),
        )


def print_gpu_info(gpu_config: GPUConfig) -> None:
    """Log detected GPU hardware and chosen optimization settings.

    Args:
        gpu_config: The resolved GPU configuration to display.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(
            f"GPU: {gpu_name}  |  {gpu_mem_gb:.1f} GB  |  "
            f"compute capability {compute_cap[0]}.{compute_cap[1]}",
            flush=True,
        )
    else:
        print("GPU: none (CPU mode)", flush=True)

    print(
        f"  flash_attention={gpu_config.flash_attention}  "
        f"bf16={gpu_config.bf16_compute}  "
        f"max_length={gpu_config.max_length}  "
        f"gradient_checkpointing={gpu_config.gradient_checkpointing}",
        flush=True,
    )


def estimate_model_billions(model_name: str) -> float:
    """Estimate model size in billions from the model name.

    Uses simple heuristics on common model name patterns.  Falls back
    to 0.6B if no pattern matches.

    Args:
        model_name: HuggingFace model identifier (e.g. ``Qwen/Qwen3-Reranker-4B``).

    Returns:
        Estimated parameter count in billions.
    """
    import re

    name_lower = model_name.lower()
    # Match patterns like "4b", "0.6b", "600m", "150m"
    match = re.search(r"(\d+\.?\d*)b(?:\b|-)", name_lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)m(?:\b|-)", name_lower)
    if match:
        return float(match.group(1)) / 1000
    return 0.6
