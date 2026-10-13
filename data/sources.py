"""Source loading for the multitask training script.

Lives outside ``model/`` and imports nothing from torch or transformers, so the
loading logic can be unit-tested and exercised on a torch-free interpreter.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from .format import TypedQuestion
from .pipeline import load_jsonl, load_synthetic

ROOT = Path(__file__).resolve().parents[1]

# Prefix for sources that omit an explicit ``path``.
DEFAULT_BENCHMARK_PREFIX = "data/benchmarks"


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def is_synthetic_source(source_name: str, source_cfg: dict | None = None) -> bool:
    """Whether a source is treated as synthetic.

    Matches the auto-detection in ``build_sampler_config``: the source name or
    its configured path contains ``"synthetic"``. Keeping one predicate keeps
    the loader and the sampler in step.
    """
    if "synthetic" in source_name:
        return True
    path_str = (source_cfg or {}).get("path") or ""
    return "synthetic" in path_str


def load_source_items(
    source_name: str,
    source_cfg: dict,
    rng: random.Random,
) -> tuple[list[TypedQuestion], list[TypedQuestion]]:
    """Load and split items for a single source.

    Args:
        source_name: Name of the source (e.g. "banking77").
        source_cfg: Source config dict with keys: path, weight, max_train, max_eval.
        rng: Random instance for deterministic sampling.

    Returns:
        (train_items, eval_items) tuple.
    """
    path_str = source_cfg.get("path")
    if not path_str and is_synthetic_source(source_name, source_cfg):
        # Synthetic data is assembled on the fly from families + variants + labels.
        items = list(load_synthetic())
        if not items:
            print(
                f"  WARNING: {source_name}: synthetic source assembled 0 items — "
                f"check {DEFAULT_BENCHMARK_PREFIX}/synthetic/",
                file=sys.stderr,
                flush=True,
            )
            return [], []
    else:
        if not path_str:
            # Default path convention
            path_str = f"{DEFAULT_BENCHMARK_PREFIX}/{source_name}.jsonl"
        path = resolve_path(path_str)

        if not path.exists():
            print(
                f"  WARNING: {path} not found, skipping {source_name}",
                file=sys.stderr,
                flush=True,
            )
            return [], []

        items = load_jsonl(path)

    # Filter to supported types
    supported = {"noul", "choice", "score"}
    items = [it for it in items if it.question.get("type") in supported]

    # Split by item.split field
    train_items = [it for it in items if it.split == "train"]
    eval_items = [it for it in items if it.split == "test"]

    # Apply max_train / max_eval caps
    max_train = source_cfg.get("max_train")
    if max_train is not None and len(train_items) > max_train:
        rng.shuffle(train_items)
        train_items = train_items[:max_train]

    max_eval = source_cfg.get("max_eval")
    if max_eval is not None and len(eval_items) > max_eval:
        rng.shuffle(eval_items)
        eval_items = eval_items[:max_eval]

    # Sources with weight: 0 are eval-only
    weight = source_cfg.get("weight", 1.0)
    if weight == 0:
        train_items = []

    return train_items, eval_items
