"""Guard the one combination that has been observed to diverge.

The readout chooses which backbone layers feed the decision heads. ``last``
takes a single pooled vector from the final layer; ``weighted`` learns a softmax
mixture over every hidden state. They are within noise of each other over short
schedules, which is why ``last`` was the default, but ``last`` has now collapsed
in two independent settings once the schedule got longer:

* **8B, full LoRA, 5 epochs** — ``last`` reached 47.51 at seed 42 and 36.43 at
  seed 123 against 62.85 and 62.99 for ``weighted`` under identical settings
  (``research/10``). A 15 to 27 point gap, reproduced on both seeds.
* **0.6B, full LoRA, 10 epochs** — ``last`` reached 0.4807 after epoch 1 and then
  sat between 0.3097 and 0.3654 for the next seven, with the training loss rising
  from 1.13 to 1.32 and never recovering. The same arm at 5 epochs reaches 54.46.

``weighted`` is at least as good at every scale that has been measured and is the
only readout that survives a long schedule, so a run that asks for ``last`` over
many epochs is very likely to be a wasted run rather than a cheaper one. The
guard warns instead of overriding: the short-schedule arms in ``research/10`` are
``last`` arms and have to stay reproducible.
"""

from __future__ import annotations

# The shortest schedule on which ``last`` has been observed to collapse, minus a
# margin: at 0.6B the divergence was visible from epoch 2 of a 10-epoch run, and
# no ``last`` arm above 5 epochs has been reported as healthy.
LONG_SCHEDULE_EPOCHS = 10


def schedule_warning(readout: str, epochs: int) -> str | None:
    """Warn when a long schedule is combined with the fragile readout.

    Args:
        readout: The resolved readout: ``last``, ``mean_last_n`` or ``weighted``.
        epochs: The number of epochs the run will train for.

    Returns:
        A warning to print, or ``None`` when the combination is not one that has
        been observed to collapse.
    """
    if readout != "last" or epochs < LONG_SCHEDULE_EPOCHS:
        return None
    return (
        f"WARNING: --readout last with {epochs} epochs. The last readout has "
        "collapsed once the schedule got longer, at both ends of the scale: 8B "
        "full LoRA reached 47.51 and 36.43 on two seeds against 62.85 and 62.99 "
        "for weighted, and 0.6B full LoRA fell from 0.4807 after epoch 1 to "
        "0.31-0.37 for the rest of a 10-epoch run. weighted is no worse at every "
        "scale measured and is the only readout that has survived a long "
        "schedule; if this run is meant to answer a question about quality "
        "rather than to reproduce a short-schedule arm, prefer --readout "
        "weighted."
    )
