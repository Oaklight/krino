"""Answer a choice question with the score head, one candidate at a time.

The choice head and the score head solve the same problem from opposite ends.
The choice head sees every option at once and normalises across them; the score
head sees one option at a time on an ordered scale and never compares two. The
backbone's own reranker is the second shape, and it beats our choice head by 14
to 21 points on JevBench (`research/13`), which is why the score-first route is
worth measuring: score each candidate on a shared ordinal scale, then take the
largest.

Nothing here trains anything. It is the evaluation bridge that makes a model
trained with ``train_multitask.py --score-form candidates`` measurable at all:
that flag converts every choice item into one score question per option, so the
choice head receives no gradient and cannot be used to score choice items. The
bridge scores a choice item the way such a model learned to answer it.

The score returned per candidate is the log-odds that the candidate sits on the
top level of the scale ("fully answers the question"). Log-odds rather than the
raw probability keeps the same convention as the native scorer's path in
``eval_calibration.py``: the values are independent across candidates and do not
sum to one, so a temperature applied to them still moves calibration without
moving the argmax.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from data.score_form import CHOICE_SCORE_LEVELS, to_score_candidates

# Keeps the log-odds finite for a candidate the model is certain about.
PROB_FLOOR = 1e-6


def top_level_log_odds(level_logits: Sequence[float]) -> float:
    """Log-odds that this candidate is on the scale's top level.

    Args:
        level_logits: The score head's logits over :data:`CHOICE_SCORE_LEVELS`.

    Returns:
        ``log p - log(1 - p)`` for the top level's probability, clamped so a
        saturated prediction stays finite.
    """
    values = [float(value) for value in level_logits]
    if not values:
        raise ValueError("a candidate needs at least one level")
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    probability = exps[-1] / total
    probability = min(max(probability, PROB_FLOOR), 1.0 - PROB_FLOOR)
    return math.log(probability) - math.log(1.0 - probability)


def candidate_questions(item: Any) -> list[Any]:
    """The per-option score questions a score-first model was trained on.

    Args:
        item: A choice ``TypedQuestion``.

    Returns:
        One score question per option, in the option order.
    """
    return to_score_candidates(item)


def score_candidates(model: Any, item: Any, device: Any = None) -> list[float]:
    """Score every option of a choice item through the score head.

    Args:
        model: A ``DecisionModel`` with a trained score head.
        item: A choice ``TypedQuestion``.
        device: Unused; kept for call sites that already hold one.

    Returns:
        Log-odds per option, in ``item``'s option order.
    """
    scores = []
    for candidate in candidate_questions(item):
        question = candidate.question
        logits = model.forward_score(
            candidate.state,
            question.get("instructions", ""),
            list(question["criteria"]),
        )
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
        if logits and isinstance(logits[0], list):
            logits = logits[0]
        scores.append(top_level_log_odds(logits))
    return scores


def predict_from_candidates(model: Any, item: Any) -> tuple[str, list[float], list[str]]:
    """The option a score-first model would choose.

    Args:
        model: A ``DecisionModel`` with a trained score head.
        item: A choice ``TypedQuestion``.

    Returns:
        ``(key, scores, keys)`` where ``key`` is the argmax option, ``scores``
        the per-option log-odds and ``keys`` the option order they align to.
        Ties go to the earlier option, matching the rest of the evaluation.
    """
    keys = list(item.question["criteria"])
    scores = score_candidates(model, item)
    if len(scores) != len(keys):
        raise ValueError(
            f"scored {len(scores)} candidates for {len(keys)} options"
        )
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return keys[best], scores, keys


def scale_name() -> tuple[str, ...]:
    """The shared ordinal scale the candidates were labelled on."""
    return CHOICE_SCORE_LEVELS
