"""The recipe hash keeps the raw marker flag (#373).

The checkpoint record stores the resolved id; the recipe keeps the raw flag so
the six ablation arms stay distinguishable.  This is the one part of the
checkpoint schema that needs ``recipe_config``, hence the torch import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model.scripts.train_residual import recipe_config


def make_args(**overrides):
    values = {
        "model": "Qwen/Qwen3-Reranker-4B",
        "option_marker": True,
        "noul_marker": False,
        "marker_token": 151651,
        "rank": 32,
        "max_length": 2048,
        "option_chunk_size": 1,
        "option_max_tokens": 64,
        "option_batch": 16,
        "train_safe_buckets": False,
        "lora_layers": None,
        "lora_rank": 16,
        "lora_alpha": 32,
        "epochs": 4,
        "patience": 1,
        "lr": 1e-3,
        "alpha_lr": 1e-2,
        "kl_weight": 1.0,
        "ce_weight": 0.2,
        "val_fraction": 0.2,
        "seed": 42,
        "answerable_head": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_the_recipe_config_still_carries_the_raw_flag(monkeypatch):
    import model.scripts.train_residual as train_residual

    monkeypatch.setattr(train_residual, "file_sha256", lambda path: "0" * 64)
    recipe = recipe_config(
        make_args(items=Path("items.jsonl"), base_probs=Path("base.jsonl"))
    )
    assert recipe["marker_token"] == 151651
