"""Share one prefix cache across option rows on hybrid (GDN) backbones.

A full-attention cache is shared by repeating its keys and values along the
batch dimension once, then cropping each suffix back off.  A Qwen3.5 cache
also holds gated-delta-net layers whose conv and recurrent states are
overwritten in place by every forward, so a suffix pass cannot be cropped
away and transformers gives those layers no ``batch_repeat_interleave``.

``fork_prefix_cache`` therefore copies the one-row prefix cache and repeats
every state of the copy, so each suffix group starts from the untouched
prefix state.  The copy is the prefix KV of the few full-attention layers
plus a fixed-size state per linear layer, independent of the suffix length.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

_STATE_DICTS = ("conv_states", "recurrent_states")


def _rows(tensor: torch.Tensor | None, rows: int) -> torch.Tensor | None:
    """A new tensor with ``rows`` copies along dim 0; never a view."""
    if tensor is None:
        return None
    return tensor.repeat_interleave(rows, dim=0)


def _fork_layer(layer: Any, rows: int) -> Any:
    fork = copy.copy(layer)
    # Shallow-copy every container so per-forward flags such as
    # ``has_previous_state`` are not shared with the prefix layer.
    for name, value in vars(layer).items():
        if isinstance(value, dict):
            setattr(fork, name, dict(value))
        elif isinstance(value, list):
            setattr(fork, name, list(value))
    for name in _STATE_DICTS:
        states = getattr(fork, name, None)
        if isinstance(states, dict):
            for index, tensor in states.items():
                states[index] = _rows(tensor, rows)
    keys = getattr(fork, "keys", None)
    if torch.is_tensor(keys) and keys.numel():
        fork.keys = _rows(keys, rows)
        fork.values = _rows(fork.values, rows)
    return fork


def fork_prefix_cache(cache: Any, rows: int) -> Any:
    """Return a ``rows``-row copy of a one-row prefix cache.

    Handles full-attention layers (``keys``/``values``), linear-attention
    layers (``conv_states``/``recurrent_states``) and layers that carry both.
    Every tensor in the copy is new memory, so the original stays untouched
    and can seed the next suffix group.
    """
    fork = copy.copy(cache)
    for name, value in vars(cache).items():
        if isinstance(value, dict):
            setattr(fork, name, dict(value))
    fork.layers = [_fork_layer(layer, rows) for layer in cache.layers]
    return fork
