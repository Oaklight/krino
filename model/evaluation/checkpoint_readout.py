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
