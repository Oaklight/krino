"""Guard the readout that has collapsed in two independent settings.

The readout chooses which backbone layers feed the decision heads. ``last`` takes
a single pooled vector from the final layer; ``weighted`` learns a softmax mixture
over every hidden state. They are within noise of each other over short schedules
on a mid-sized backbone, which is why ``last`` was the default — but ``last`` has
now collapsed twice, and for two different reasons:

* **8B, full LoRA, five epochs** — ``last`` reached 47.51% at seed 42 and 36.43%
  at seed 123 against 62.85% and 62.99% for ``weighted`` under identical settings
  (``research/10``). This is not about schedule length: it happened at the
  project's standard five epochs, so it is a scale effect (or scale x LoRA).
* **0.6B, full LoRA, ten epochs** — ``last`` reached 48.07% after epoch 1 and then
  sat between 30.97% and 36.54% for the rest of the run, with the training loss
  rising from 1.13 to 1.32 and never recovering. The same arm at five epochs
  reaches 54.46%. This one is about schedule length. The per-epoch history is
  committed as ``model/experiments/arch_06b_loraAll_last_10ep_s123_epochs.json``
  so those numbers can be checked from the checkout.

``weighted`` is at least as good at every scale that has been measured and is the
only readout that survived either failure, so it is now the default. This guard
stays for the case where someone asks for ``last`` explicitly: it warns and never
overrides, because the short-schedule arms in ``research/10`` are ``last`` arms and
have to stay reproducible with ``--readout last``.
"""

from __future__ import annotations

import re

# Schedules at or above this length are the ones the 0.6B arm failed on.
LONG_SCHEDULE_EPOCHS = 10

# Backbones at or above this size failed at the ordinary five-epoch length.
FRAGILE_PARAMETER_BILLIONS = 8.0


def backbone_billions(model_name: str | None) -> float | None:
    """Backbone size in billions of parameters, read from a model name.

    ``Qwen/Qwen3-Reranker-8B`` -> 8.0, ``...-0.6B`` -> 0.6. Returns ``None`` when
    the name carries no size, so an unrecognised name never triggers the scale
    rule by accident.
    """
    if not model_name:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", str(model_name))
    return float(match.group(1)) if match else None


def schedule_warning(
    readout: str, epochs: int, model_name: str | None = None
) -> str | None:
    """Warn when the readout is combined with a setting it has collapsed in.

    Args:
        readout: The resolved readout: ``last``, ``mean_last_n`` or ``weighted``.
        epochs: The number of epochs the run will train for.
        model_name: The backbone name, used only to read its size.

    Returns:
        A one-paragraph warning to print, or ``None`` when the combination has
        not been observed to collapse.
    """
    if readout != "last":
        return None
    billions = backbone_billions(model_name)
    large = billions is not None and billions >= FRAGILE_PARAMETER_BILLIONS
    long_schedule = epochs >= LONG_SCHEDULE_EPOCHS
    if not (large or long_schedule):
        return None
    where = (
        f"{billions:g}B and {epochs} epochs"
        if large and long_schedule
        else f"{billions:g}B" if large
        else f"{epochs} epochs"
    )
    return (
        f"WARNING: --readout last with {where}. last has collapsed twice: at 8B "
        "full LoRA it reached 47.51% and 36.43% on two seeds against 62.85% and "
        "62.99% for weighted, at the ordinary five epochs; and at 0.6B full LoRA "
        "over ten epochs it fell from 48.07% after epoch 1 to 31.0-36.5% for the "
        "rest of the run. weighted is no worse at every measured scale. Pass "
        "--readout weighted unless this run is deliberately reproducing a "
        "short-schedule last arm."
    )
