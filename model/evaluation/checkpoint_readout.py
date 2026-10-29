"""Read the readout a checkpoint was trained with, from its tensor names alone.

The readout decides whether the heads see the last hidden state or a learned
mixture over every layer. Loading a checkpoint through the wrong one does not
fail loudly — ``load_heads`` is not strict, so a weighted checkpoint evaluated as
``last`` drops its mixture and scores a different network. The reverse (a ``last``
checkpoint evaluated as ``weighted``) leaves the mixture at initialisation and is
just as quiet.

Only one direction is recoverable from the names: the weighted readout carries
``readout.*`` tensors and the others do not. So this reports ``"weighted"`` when
it can prove it and ``None`` otherwise, rather than guessing between ``last`` and
``mean_last_n``, which look identical in a checkpoint.
"""

from __future__ import annotations

from typing import Iterable


def infer_readout(state_keys: Iterable[str]) -> str | None:
    """The readout a checkpoint was trained with, when its tensors say so.

    Args:
        state_keys: The keys of the checkpoint's state dict.

    Returns:
        ``"weighted"`` when the checkpoint carries the learned layer mixture, and
        ``None`` when it does not — ``last`` and ``mean_last_n`` are
        indistinguishable from the keys, so neither is guessed.
    """
    for key in state_keys:
        name = str(key)
        if name.startswith("readout.") and "layer_logits" in name:
            return "weighted"
    return None


def infer_choice_mode(state_keys: Iterable[str]) -> str | None:
    """The choice head shape a checkpoint was trained with.

    A cross-encoder checkpoint carries ``choice_head.norm.*`` and
    ``choice_head.score.*``; the default attention head carries
    ``choice_head.attention.*`` instead. Loading one through the other either
    fails or, worse, leaves the real head at initialisation while reporting a
    plausible number, so evaluation reads this from the tensors.

    Args:
        state_keys: The keys of the checkpoint's state dict.

    Returns:
        ``"cross_encoder"`` when the checkpoint has the linear read-out,
        ``"tower"`` when it has the attention head, and ``None`` when it has
        neither (a checkpoint that never trained a choice head).
    """
    keys = [str(key) for key in state_keys]
    if any(name.startswith("choice_head.norm.") or name.startswith("choice_head.score.") for name in keys):
        return "cross_encoder"
    if any(name.startswith("choice_head.attention.") for name in keys):
        return "tower"
    return None
