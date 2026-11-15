"""The residual checkpoint records the marker it trained against (#373).

The marker id is part of the checkpoint's own config, not the recipe hash, so
a consumer can rebuild the arm without falling back to the tokenizer default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model.scripts.train_residual import checkpoint_config
from model.src.marker import resolve_checkpoint_marker


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


class TestCheckpointConfig:
    def test_the_checkpoint_records_its_marker_token(self):
        saved = checkpoint_config(make_args(), 0.0)
        assert saved["marker_token"] == 151651

    def test_the_saved_marker_round_trips_through_the_consumer(self):
        saved = checkpoint_config(make_args(), 0.0)
        assert resolve_checkpoint_marker(saved) == 151651

    def test_a_non_marker_checkpoint_records_no_marker(self):
        saved = checkpoint_config(
            make_args(option_marker=False, marker_token=None), 0.0
        )
        assert saved["marker_token"] is None
        assert resolve_checkpoint_marker(saved) is None

    def test_the_recipe_config_still_carries_the_marker(self, monkeypatch):
        # The recipe hash must keep varying with the marker: that is how the
        # six arms stay distinguishable.  The checkpoint record is separate.
        import model.scripts.train_residual as train_residual

        monkeypatch.setattr(train_residual, "file_sha256", lambda path: "0" * 64)
        recipe = train_residual.recipe_config(
            make_args(items=Path("items.jsonl"), base_probs=Path("base.jsonl"))
        )
        assert recipe["marker_token"] == 151651
