"""Decide which level a score head's distribution predicts.

A score head emits a distribution over ordered levels, and something has to turn
it into one level. The obvious choice -- the argmax -- is the mode, and the mode
is only optimal when every mistake costs the same. QWK's disagreement weights are
``(i - j) ** 2``: a one-level miss is cheap and a three-level miss is nine times
cheaper than it is costly. Under squared error the Bayes-optimal decision is the
posterior mean, not the mode, and on the five-level score scale the difference is
worth 0.024 QWK.

Three decoders, all deterministic:

* ``argmax`` -- the mode. What the harness reports today.
* ``ordinal_median`` -- the smallest level whose cumulative probability reaches
  a half. Minimises absolute error, so it is the right rule when the cost of a
  miss grows with distance but not with its square.
* ``expected_level`` -- the posterior mean, rounded and clamped. Minimises
  squared error, which is QWK's cost structure.

Nothing here knows what a model is; the functions take a logit vector and return
an index, so the arithmetic can be tested without torch.
"""

from __future__ import annotations

import math
from typing import Sequence


def _probabilities(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax over one logit vector."""
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def decode_argmax(logits: Sequence[float]) -> int:
    """The mode, ties broken toward the lower level."""
    if not logits:
        raise ValueError("a logit vector is required")
    return max(range(len(logits)), key=lambda index: (logits[index], -index))


def decode_ordinal_median(logits: Sequence[float]) -> int:
    """The smallest level whose cumulative probability reaches one half."""
    probabilities = _probabilities(logits)
    if not probabilities:
        raise ValueError("a logit vector is required")
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= 0.5:
            return index
    return len(probabilities) - 1


def decode_expected_level(logits: Sequence[float], bias: float = 0.0) -> int:
    """The posterior mean level, rounded and clamped to the scale.

    ``bias`` shifts the mean before rounding. It defaults to zero, which is the
    Bayes rule under squared error; a non-zero bias is a fitted parameter and
    belongs in a report only if it was fitted out of sample.

    Args:
        logits: One score per level, in level order.
        bias: Added to the mean before rounding.

    Returns:
        The chosen level index.
    """
    probabilities = _probabilities(logits)
    if not probabilities:
        raise ValueError("a logit vector is required")
    mean = sum(index * probability for index, probability in enumerate(probabilities))
    return max(0, min(len(probabilities) - 1, int(round(mean + bias))))


DECODERS = {
    "argmax": decode_argmax,
    "ordinal_median": decode_ordinal_median,
    "expected_level": decode_expected_level,
}
