"""Score-first formulation of typed questions.

The score question is the most fundamental of Krino's three question types:
a ``noul`` question is a score question over ``["false", "true"]``, and a
``choice`` question can be decomposed into one score question per candidate
option.  This module implements the conversions in both directions so that
datasets and training can be built in score form and projected back to the
answer shape a consumer expects.

Score convention
----------------
A score question carries ``question["criteria"]`` as an ordered list of level
descriptions, and its label is the *expected level index* in
``[0, len(criteria) - 1]``.  That is the convention used by the rest of the
repository: ``model/src/decision_model.py`` reports
``sum(i * p for i, p in enumerate(probs))``, and
``model/training/supervised.py`` interpolates float labels between adjacent
integer levels.

Conversions
-----------
noul <-> score
    The two levels are ``["false", "true"]``.  The expected level index is
    ``0 * P(false) + 1 * P(true) = P(true)``, so the score is exactly the
    probability that the answer is true.  Converting back thresholds the
    normalized score at 0.5 (see :func:`score_to_noul`).

choice <-> score
    Choice options are *unordered*, which the score schema cannot represent
    faithfully.  Two encodings are provided:

    ``to_score_candidates`` (preferred)
        One score question per candidate option, all on the shared ordinal
        scale :data:`CHOICE_SCORE_LEVELS` ("how well does this candidate
        answer the question?").  Each candidate is scored independently and
        ``argmax`` over the per-candidate scores recovers the choice.

    ``to_score`` (lossy, single question)
        The option list is treated as an ordered level scale and the label is
        the index of the gold option.  A hard label round-trips exactly, but
        the ordering is spurious: the expected index of a spread categorical
        distribution can land on an option that has no probability mass.  Use
        ``to_score_candidates`` when calibrated choice probabilities matter.

score <-> score
    Identity.

All functions are pure and use the standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .format import TypedQuestion

__all__ = [
    "CHOICE_SCORE_LEVELS",
    "NOUL_SCORE_LEVELS",
    "ScorePrediction",
    "convert_training_items",
    "from_score",
    "round_trip",
    "score_to_choice",
    "score_to_noul",
    "to_score",
    "to_score_candidates",
]


#: Ordered levels used when a noul question is written as a score question.
#: The expected level index over these levels is exactly ``P(true)``.
NOUL_SCORE_LEVELS: tuple[str, str] = ("false", "true")

#: Shared ordinal scale used to score each choice candidate independently.
#: The first level is the worst answer and the last level is the best.
CHOICE_SCORE_LEVELS: tuple[str, ...] = (
    "does not answer the question",
    "barely answers the question",
    "partially answers the question",
    "mostly answers the question",
    "fully answers the question",
)

#: A score-form prediction.  A float is an expected level index.  For choice
#: questions a mapping of option key to score, or a sequence of scores aligned
#: with the option order, is also accepted by :func:`from_score`.
ScorePrediction = float | Mapping[str, float] | Sequence[float]


def to_score(question: TypedQuestion) -> TypedQuestion:
    """Return the score-form representation of ``question``.

    ``noul`` questions become a two-level score question over
    ``["false", "true"]`` whose label is ``P(true)``.  ``choice`` questions
    become a single score question whose levels are the option texts, in the
    original order, and whose label is the gold option's index.  That
    single-question choice encoding is lossy -- see the module docstring and
    :func:`to_score_candidates`.  ``score`` questions are returned unchanged.

    Teacher probabilities are remapped to level-index keys where the mapping
    is unambiguous (``"0"``, ``"1"``, ...).  Ambiguous or augmented
    distributions are dropped rather than silently reinterpreted.  Augmented
    noul questions (``question["augmented_options"]``) are reduced to their
    boolean form.

    Args:
        question: Source question of type ``noul``, ``choice``, or ``score``.

    Returns:
        A score-form ``TypedQuestion``.  For ``score`` inputs this is the same
        object.

    Raises:
        ValueError: If the question type is unknown, a choice question has no
            options, or its label is not one of the option keys.
    """
    q_type = _question_type(question)

    if q_type == "score":
        return question

    if q_type == "noul":
        return TypedQuestion(
            id=question.id,
            state=question.state,
            question={
                "type": "score",
                "instructions": question.question.get("instructions", ""),
                "criteria": list(NOUL_SCORE_LEVELS),
                "score_form": {
                    "source_type": "noul",
                    "encoding": "boolean-as-score",
                    "parent_id": question.id,
                },
            },
            label=1.0 if _as_bool(question.label) else 0.0,
            source=question.source,
            split=question.split,
            group=question.group,
            teacher_probs=_noul_teacher_probs(question.teacher_probs),
        )

    if q_type == "choice":
        criteria = _choice_criteria(question)
        keys = list(criteria)
        if question.label not in criteria:
            raise ValueError(
                f"choice label {question.label!r} is not one of the option keys {keys!r}"
            )
        return TypedQuestion(
            id=question.id,
            state=question.state,
            question={
                "type": "score",
                "instructions": question.question.get("instructions", ""),
                "criteria": _option_texts(criteria),
                "score_form": {
                    "source_type": "choice",
                    "encoding": "option-as-level",
                    "option_keys": keys,
                    "parent_id": question.id,
                },
            },
            label=float(keys.index(question.label)),
            source=question.source,
            split=question.split,
            group=question.group,
            teacher_probs=_choice_teacher_probs(question.teacher_probs, keys),
        )

    raise ValueError(f"unknown question type: {q_type!r}")


def to_score_candidates(question: TypedQuestion) -> list[TypedQuestion]:
    """Return per-candidate score questions for ``question``.

    For ``choice`` questions this returns one score question per option, all
    on the shared ordinal scale :data:`CHOICE_SCORE_LEVELS`.  The candidate is
    named in the instructions and in ``question["score_form"]["candidate"]``.
    The gold option is labelled with the top level
    (``len(CHOICE_SCORE_LEVELS) - 1``) and every other option with the bottom
    level (``0.0``).  This preserves the ``argmax`` for a hard label but
    discards any graded quality of the distractors, and it does not preserve
    a categorical probability distribution over options.

    For ``noul`` and ``score`` questions this returns
    ``[to_score(question)]``.

    Teacher probabilities are dropped.  A categorical distribution over the
    options does not map onto independent per-candidate ordinal scores, so
    unlike :func:`to_score` there is no faithful remapping to keep.

    Options whose text repeats are disambiguated in the prompt.  A
    deterministic scorer would otherwise give two identical prompts identical
    scores, and the ``argmax`` tie-break would return the first key even when
    the gold answer is the other one -- hellaswag endings repeat often enough
    for this to matter.

    Args:
        question: Source question of type ``noul``, ``choice``, or ``score``.

    Returns:
        One score question per choice option, in the original criteria order;
        a single-element list for the other question types.

    Raises:
        ValueError: If the question type is unknown, a choice question has no
            options, or its label is not one of the option keys.
    """
    q_type = _question_type(question)

    if q_type != "choice":
        return [to_score(question)]

    criteria = _choice_criteria(question)
    keys = list(criteria)
    if question.label not in criteria:
        raise ValueError(
            f"choice label {question.label!r} is not one of the option keys {keys!r}"
        )

    instructions = question.question.get("instructions", "").strip()
    top_level = float(len(CHOICE_SCORE_LEVELS) - 1)
    texts = {key: criteria[key] or key for key in keys}
    repeated = {
        text for text in texts.values() if list(texts.values()).count(text) > 1
    }
    candidates: list[TypedQuestion] = []
    for key in keys:
        text = texts[key]
        parts = [instructions]
        heading = f"Candidate answer ({key})" if text in repeated else "Candidate answer"
        parts.append(f"{heading}: {text}.")
        parts.append("How well does this candidate answer the question?")
        candidates.append(
            TypedQuestion(
                id=f"{question.id}#{key}",
                state=question.state,
                question={
                    "type": "score",
                    "instructions": " ".join(part for part in parts if part),
                    "criteria": list(CHOICE_SCORE_LEVELS),
                    "score_form": {
                        "source_type": "choice",
                        "encoding": "per-candidate",
                        "candidate": key,
                        "candidate_text": text,
                        "option_keys": keys,
                        "parent_id": question.id,
                    },
                },
                label=top_level if key == question.label else 0.0,
                source=question.source,
                split=question.split,
                group=question.group,
                teacher_probs=None,
            )
        )
    return candidates


def convert_training_items(
    items: list[TypedQuestion], mode: str
) -> list[TypedQuestion]:
    """Convert a training set into score form.

    Only training items are converted.  Evaluation stays in the original task
    shape so accuracy remains comparable between a score-form run and an
    ordinary one; the heads are shared, so a model trained on score questions
    still answers noul and choice items through their usual forward paths.

    Args:
        items: Training items in their original types.
        mode: ``"single"`` gives one score question per item.  ``"candidates"``
            expands each choice item into one score question per candidate,
            which grows the set by roughly the average option count.

    Returns:
        The converted items.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode == "single":
        return [to_score(item) for item in items]
    if mode == "candidates":
        converted: list[TypedQuestion] = []
        for item in items:
            converted.extend(to_score_candidates(item))
        return converted
    raise ValueError(
        f"unknown score-form mode {mode!r}; expected 'single' or 'candidates'"
    )


def from_score(prediction: ScorePrediction, question: TypedQuestion) -> dict[str, Any]:
    """Convert a score-form prediction back to the original answer shape.

    Args:
        prediction: Score-form prediction.  A float is an expected level
            index.  For choice questions a mapping of option key to score, or
            a sequence of scores in the original option order, is also
            accepted; the winner is then chosen by ``argmax`` and the reported
            probabilities come from a softmax over the scores.
        question: The original question whose type determines the output
            shape.

    Returns:
        ``{"noul": p}`` for noul questions, where ``p`` is ``P(true)``
        clamped to ``[0, 1]``; ``{"choice": label, "probabilities": {...}}``
        for choice questions; and ``{"score": value}`` for score questions.

        A scalar choice prediction carries no distributional information, so
        its probabilities are one-hot at the bucketized option.  Per-candidate
        scores produce a soft distribution via softmax; it is a ranking-derived
        score, not a calibrated probability.

    Raises:
        TypeError: If the prediction shape is not valid for the question type.
        ValueError: If the question type is unknown, per-candidate scores do
            not match the option keys, or a prediction is not finite.
    """
    q_type = _question_type(question)

    if q_type == "noul":
        score = _scalar_prediction(prediction, "noul")
        return {"noul": _clamp(score, 0.0, 1.0)}

    if q_type == "score":
        score = _scalar_prediction(prediction, "score")
        return {"score": score}

    if q_type == "choice":
        criteria = _choice_criteria(question)
        keys = list(criteria)
        if isinstance(prediction, Mapping):
            scores = _mapping_scores(prediction, keys)
            return _choice_from_scores(scores, keys)
        if _is_non_string_sequence(prediction):
            values = [
                _as_float(value, f"prediction[{index}]")
                for index, value in enumerate(prediction)
            ]
            if len(values) != len(keys):
                raise ValueError(
                    f"expected {len(keys)} candidate scores, got {len(values)}"
                )
            return _choice_from_scores(dict(zip(keys, values)), keys)
        score = _scalar_prediction(prediction, "choice")
        return _choice_from_scalar(score, question)

    raise ValueError(f"unknown question type: {q_type!r}")


def score_to_noul(
    prediction: float,
    n_levels: int = len(NOUL_SCORE_LEVELS),
    threshold: float = 0.5,
) -> bool:
    """Threshold a score-form prediction into a noul answer.

    The expected level index is rescaled to ``[0, 1]`` by dividing by
    ``n_levels - 1`` and compared with ``threshold`` using a strict
    greater-than, matching ``model/evaluation/accuracy.py``.

    Args:
        prediction: Expected level index.
        n_levels: Number of levels in the score question.  Must be at least 2.
        threshold: Decision threshold on the normalized scale, in ``[0, 1]``.

    Returns:
        True when the normalized score is strictly above ``threshold``.

    Raises:
        ValueError: If ``n_levels < 2``, ``threshold`` is outside ``[0, 1]``,
            or ``prediction`` is not finite.
        TypeError: If ``prediction`` or ``threshold`` is not a real number.
    """
    score = _as_float(prediction, "prediction")
    threshold_value = _as_float(threshold, "threshold")
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2 to threshold a score, got {n_levels}")
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold_value!r}")
    normalized = score / (n_levels - 1)
    return normalized > threshold_value


def score_to_choice(prediction: float, question: TypedQuestion) -> str:
    """Bucketize a score-form prediction into a choice label.

    This is the inverse of the single-question ``choice -> score`` encoding
    produced by :func:`to_score`, where the option list doubles as an ordered
    level scale.  The expected index is rounded to the nearest level (half-up)
    and clamped to the valid range.

    Args:
        prediction: Expected level index on the option-as-level scale.
        question: The original choice question, which supplies the option keys.

    Returns:
        The option key of the nearest level.

    Raises:
        ValueError: If ``question`` is not a choice question, has no options,
            or ``prediction`` is not finite.
        TypeError: If ``prediction`` is not a real number.
    """
    if _question_type(question) != "choice":
        raise ValueError(
            f"score_to_choice requires a choice question, got {_question_type(question)!r}"
        )
    keys = list(_choice_criteria(question))
    score = _as_float(prediction, "prediction")
    index = _nearest_level(score, len(keys))
    return keys[index]


def round_trip(question: TypedQuestion) -> dict[str, Any]:
    """Convert to score form and back, reporting fidelity.

    The round trip uses :func:`to_score` and :func:`from_score` with the
    score-form label as the prediction, i.e. it checks that the *gold* answer
    survives the conversion.  It does not check that arbitrary soft
    predictions survive -- for choice questions they do not (see the module
    docstring).

    Args:
        question: Source question of type ``noul``, ``choice``, or ``score``.

    Returns:
        A dict with keys:

        - ``type``: the original question type.
        - ``score_question``: the :func:`to_score` result.
        - ``recovered``: the :func:`from_score` answer dict.
        - ``recovered_label``: the recovered answer in the original label
          shape (bool for noul, option key for choice, float for score).
        - ``exact``: whether ``recovered_label`` equals the original label.
        - ``lossy``: whether the conversion loses information.  True for
          choice, whose options are unordered.
        - ``note``: a human-readable explanation when ``lossy`` is True,
          otherwise ``None``.

    Raises:
        ValueError: If the question type is unknown or the conversion is not
            defined for the question's fields.
    """
    q_type = _question_type(question)
    score_question = to_score(question)
    prediction = _as_float(score_question.label, "score-form label")
    recovered = from_score(prediction, question)

    if q_type == "noul":
        recovered_label: Any = score_to_noul(recovered["noul"])
        original_label: Any = _as_bool(question.label)
        lossy = False
        note: str | None = None
    elif q_type == "choice":
        recovered_label = recovered["choice"]
        original_label = question.label
        lossy = True
        note = (
            "choice options are unordered; to_score encodes them as an ordinal "
            "level scale, so a soft distribution can bucketize to an option with "
            "no probability mass. Use to_score_candidates for per-candidate scoring."
        )
    elif q_type == "score":
        recovered_label = recovered["score"]
        original_label = _as_float(question.label, "label")
        lossy = False
        note = None
    else:  # pragma: no cover - to_score already raised
        raise ValueError(f"unknown question type: {q_type!r}")

    return {
        "type": q_type,
        "score_question": score_question,
        "recovered": recovered,
        "recovered_label": recovered_label,
        "exact": recovered_label == original_label,
        "lossy": lossy,
        "note": note,
    }


def _question_type(question: TypedQuestion) -> str:
    """Return the question type, defaulting to ``noul`` like the training code."""
    return question.question.get("type", "noul")


def _as_bool(value: Any) -> bool:
    """Interpret a noul label as a bool, accepting the repo's string aliases."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    raise ValueError(f"cannot interpret noul label as a bool: {value!r}")


def _as_float(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def _nearest_level(value: float, n_levels: int) -> int:
    """Round an expected level index to the nearest level (half-up) and clamp."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    index = math.floor(value + 0.5)
    return max(0, min(n_levels - 1, index))


def _softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax over finite scores."""
    if not scores:
        return []
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _is_non_string_sequence(value: Any) -> bool:
    """True for sequences that are not strings/bytes."""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _choice_criteria(question: TypedQuestion) -> Mapping[str, str]:
    """Return and validate a choice question's option mapping."""
    criteria = question.question.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError("choice question must have a non-empty 'criteria' mapping")
    return criteria


def _option_texts(criteria: Mapping[str, str]) -> list[str]:
    """Return option texts in criteria order, falling back to keys."""
    return [text if text else key for key, text in criteria.items()]


def _noul_teacher_probs(
    teacher_probs: dict[str, float] | None,
) -> dict[str, float] | None:
    """Map noul teacher probabilities onto the two score levels.

    Only the unambiguous ``{"true", "false"}`` and ``{"noul"}`` shapes are
    converted; augmented multi-option distributions are dropped.
    """
    if not teacher_probs:
        return None
    if set(teacher_probs) == {"true", "false"}:
        p_true = _as_float(teacher_probs["true"], "teacher_probs['true']")
    elif set(teacher_probs) == {"noul"}:
        p_true = _as_float(teacher_probs["noul"], "teacher_probs['noul']")
    else:
        return None
    p_true = _clamp(p_true, 0.0, 1.0)
    return {"0": 1.0 - p_true, "1": p_true}


def _choice_teacher_probs(
    teacher_probs: dict[str, float] | None,
    keys: list[str],
) -> dict[str, float] | None:
    """Map choice teacher probabilities onto level-index keys.

    The distribution is copied as-is (no renormalization) and only when its
    keys exactly match the option keys.
    """
    if not teacher_probs or set(teacher_probs) != set(keys):
        return None
    return {
        str(index): _as_float(teacher_probs[key], f"teacher_probs[{key!r}]")
        for index, key in enumerate(keys)
    }


def _scalar_prediction(prediction: Any, q_type: str) -> float:
    """Validate that a prediction is a scalar score."""
    if isinstance(prediction, Mapping) or _is_non_string_sequence(prediction):
        raise TypeError(
            f"{q_type} questions expect a scalar score prediction, "
            f"got {type(prediction).__name__}"
        )
    return _as_float(prediction, "prediction")


def _mapping_scores(prediction: Mapping[str, float], keys: list[str]) -> dict[str, float]:
    """Validate per-candidate scores against the option keys."""
    missing = [key for key in keys if key not in prediction]
    extra = [key for key in prediction if key not in keys]
    if missing or extra:
        raise ValueError(
            "candidate score keys do not match the option keys; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return {key: _as_float(prediction[key], f"score[{key!r}]") for key in keys}


def _choice_from_scores(scores: dict[str, float], keys: list[str]) -> dict[str, Any]:
    """Pick the argmax option and report softmax probabilities."""
    label = max(keys, key=lambda key: scores[key])
    probabilities = _softmax([scores[key] for key in keys])
    return {"choice": label, "probabilities": dict(zip(keys, probabilities))}


def _choice_from_scalar(score: float, question: TypedQuestion) -> dict[str, Any]:
    """Bucketize a scalar score and report a one-hot distribution."""
    keys = list(_choice_criteria(question))
    label = score_to_choice(score, question)
    return {
        "choice": label,
        "probabilities": {key: 1.0 if key == label else 0.0 for key in keys},
    }
