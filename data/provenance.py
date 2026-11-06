"""Provenance tiers and the hard-label eligibility rule.

Every training record carries one tier:

- ``gold``      the label is the source dataset's label, or an authored label
                in a generated dataset.  It is not a model prediction on the
                record itself.
- ``execution`` the label is program or engine output: a recorded agent
                trajectory, or a label constructed by a script.
- ``teacher``   the label is a Jev prediction.  Soft target only, never CE.

See ``research/25-corpus-inventory.md`` for the per-source decision record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GOLD = "gold"
EXECUTION = "execution"
TEACHER = "teacher"
TIERS = (GOLD, EXECUTION, TEACHER)

# Benchmark conversions whose labels come from the source dataset.
DATASET_SOURCES = frozenset(
    {
        "2wiki_long_choice",
        "agnews",
        "anli",
        "arc",
        "banking77",
        "boolq",
        "codesearchnet",
        "commonsenseqa",
        "contractnli",
        "fever",
        "hellaswag",
        "logiqa",
        "longbench_long_choice",
        "mednli",
        "mmlu",
        "mmlu_pro_short_reasoning_choice",
        "mnli",
        "multirc",
        "piqa",
        "race",
        "race_short_passage_choice",
        "sst2",
        "sst5",
        "stsb",
        "swag",
        "tabfact",
        "twowiki_long_choice",
        "winogrande",
        "yelp",
        "hotpotqa_long_choice",
    }
)

# Authored community sets: labels were written for the dataset, not predicted
# by a model on the record.
AUTHORED_SOURCES = frozenset({"typed_decisions", "openjev_authored144"})

SYNTHETIC_SOURCES = frozenset({"synthetic"})

_ANSWER_REASONS = frozenset(
    {"insufficient", "out_of_scope", "none_of_these", "no_applicable_tool"}
)


@dataclass(frozen=True)
class Provenance:
    tier: str
    origin: str
    reason: str


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def classify(item: Any, corpus: str | None = None) -> Provenance:
    """Return the tier, label origin, and the reason for one record.

    ``corpus`` names the file or corpus the record was read from.  It is
    required when the record's own ``source`` cannot identify the construction;
    the abstention files, for example, carry the donor dataset in ``source``.
    """
    declared = _get(item, "provenance")
    if declared is not None:
        if declared not in TIERS:
            raise ValueError(f"unknown provenance tier: {declared!r}")
        return Provenance(str(declared), "declared", "record already carries a tier")

    source = str(_get(item, "source") or "")
    item_id = str(_get(item, "id") or "")
    corpus_name = (corpus or "").rsplit("/", 1)[-1].removesuffix(".jsonl")

    if source.startswith("apigen") or corpus_name.startswith("apigen"):
        return Provenance(
            EXECUTION,
            "trajectory",
            "label is the first recorded function_call or a program "
            "classification of the first assistant turn",
        )

    reason = _get(item, "reason")
    if (
        "-abstain-" in item_id
        or corpus_name.startswith("abstention")
        or (reason in _ANSWER_REASONS and _get(item, "answerable") is False)
    ):
        return Provenance(
            EXECUTION,
            "constructed",
            "label is constructed by model/scripts/build_abstention_sets.py",
        )

    if source in SYNTHETIC_SOURCES or corpus_name.startswith("synthetic"):
        return Provenance(
            GOLD,
            "synthetic_authored",
            "label is authored into the synthetic dataset, not a Jev prediction",
        )

    if source in AUTHORED_SOURCES:
        return Provenance(
            GOLD,
            "community_authored",
            "label is authored in a community dataset, not a Jev prediction",
        )

    if source in DATASET_SOURCES:
        return Provenance(GOLD, "dataset", "label is the source dataset's label")

    raise ValueError(f"cannot assign provenance to source={source!r} corpus={corpus!r}")


def tag(item: Any, corpus: str | None = None) -> Provenance:
    """Classify and write ``item.provenance`` in place when the item is mutable.

    Pass ``corpus`` whenever the record came from a constructed corpus.  Without
    it, a record whose ``source`` names a donor dataset (the abstention files)
    classifies by the dataset and the wrong tier is stamped permanently.
    """
    provenance = classify(item, corpus)
    if isinstance(item, dict):
        item.setdefault("provenance", provenance.tier)
    elif hasattr(item, "provenance"):
        if item.provenance is None:
            item.provenance = provenance.tier
    return provenance


def normalise_label(item: Any) -> Any:
    """Coerce the two label encodings the corpora contain.

    ``"true"``/``"false"`` -> ``bool``; a numeric string -> ``float``.  Other
    labels are returned unchanged.
    """
    question = _get(item, "question") or {}
    q_type = question.get("type") if isinstance(question, dict) else None
    label = _get(item, "label")
    if q_type == "noul" and isinstance(label, str):
        lowered = label.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if q_type == "score" and isinstance(label, str):
        return float(label)
    return label


def may_drive_hard_label(item: Any, corpus: str | None = None) -> bool:
    """Whether a record may contribute to CE/BCE/ordinal loss.

    Fails closed: an unclassifiable record, a ``teacher`` record, an abstention
    record, or a label that does not validate returns False.
    """
    if _get(item, "answerable") is False:
        return False
    try:
        tier = classify(item, corpus).tier
    except ValueError:
        return False
    if tier not in (GOLD, EXECUTION):
        return False

    question = _get(item, "question")
    if not isinstance(question, dict):
        return False
    q_type = question.get("type")
    if _get(item, "label") is None:
        return False
    try:
        label = normalise_label(item)
    except (TypeError, ValueError):
        return False

    if q_type == "noul":
        return isinstance(label, bool)
    if q_type == "choice":
        criteria = question.get("criteria")
        return isinstance(criteria, dict) and label in criteria
    if q_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            return False
        return 0.0 <= float(label) <= len(criteria) - 1
    return False
