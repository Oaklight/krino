"""Typed-question data format definitions.

Every benchmark is converted into this unified format:

    {
        "id": "banking77-train-0042",
        "state": "I want to close my account",
        "question": {
            "type": "choice",
            "instructions": "Which intent category does this message belong to?",
            "criteria": {"close_account": "...", "cancel_transfer": "...", ...}
        },
        "label": "close_account",
        "source": "banking77",
        "split": "train",
        "group": "group_id_for_entity_level_splitting"
    }

For noul questions, label is a boolean (True/False).
For score questions, label is a float.
For choice questions, label is the option key string.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TypedQuestion:
    id: str
    state: str
    question: dict[str, Any]
    label: Any
    source: str
    split: str
    group: str | None = None
    teacher_probs: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def noul(
        id: str,
        state: str,
        instructions: str,
        label: bool,
        source: str,
        split: str,
        group: str | None = None,
        criteria: dict[str, str] | None = None,
    ) -> TypedQuestion:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = criteria
        return TypedQuestion(id=id, state=state, question=question, label=label, source=source, split=split, group=group)

    @staticmethod
    def choice(
        id: str,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        label: str,
        source: str,
        split: str,
        group: str | None = None,
    ) -> TypedQuestion:
        return TypedQuestion(
            id=id,
            state=state,
            question={"type": "choice", "instructions": instructions, "criteria": criteria},
            label=label,
            source=source,
            split=split,
            group=group,
        )

    @staticmethod
    def score(
        id: str,
        state: str,
        instructions: str,
        criteria: list[str],
        label: float,
        source: str,
        split: str,
        group: str | None = None,
    ) -> TypedQuestion:
        return TypedQuestion(
            id=id,
            state=state,
            question={"type": "score", "instructions": instructions, "criteria": criteria},
            label=label,
            source=source,
            split=split,
            group=group,
        )
