"""The detailed eval breakdown must not merge question types within a source.

``by_source`` mixes every type a source contributes and only remembers the
first type it saw, so a multi-type source like ``synthetic`` gets mislabelled.
``by_source_type`` is the authoritative per-(source, type) view.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data.format import TypedQuestion
from model.training.supervised import eval_epoch_detailed


class _ConstantModel(torch.nn.Module):
    """Predicts a fixed class per question type, ignoring the input."""

    def __init__(self, noul_logit: float, choice_best: int, score_best: int):
        super().__init__()
        self.noul_logit = noul_logit
        self.choice_best = choice_best
        self.score_best = score_best
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward_noul(self, state, instructions):
        return torch.tensor([[self.noul_logit]])

    def forward_choice(self, state, instructions, option_texts):
        logits = torch.zeros(1, len(option_texts))
        logits[0, self.choice_best] = 10.0
        return logits

    def forward_score(self, state, instructions, levels):
        logits = torch.zeros(1, len(levels))
        logits[0, self.score_best] = 10.0
        return logits


def _noul(source: str, item_id: str, label: bool) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "noul", "instructions": "is it?"},
        label=label,
        source=source,
        split="test",
    )


def _choice(source: str, item_id: str, label: str) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        label=label,
        source=source,
        split="test",
    )


def _score(source: str, item_id: str, label: float) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "score", "instructions": "rate", "criteria": ["1", "2", "3"]},
        label=label,
        source=source,
        split="test",
    )


def test_multi_type_source_is_split_by_type():
    # Always-true noul head, always-"a" choice head, always-level-0 score head.
    model = _ConstantModel(noul_logit=10.0, choice_best=0, score_best=0)

    items = [
        # noul: 2 right, 1 wrong
        _noul("synthetic", "s-noul-1", True),
        _noul("synthetic", "s-noul-2", True),
        _noul("synthetic", "s-noul-3", False),
        # choice: 1 right, 1 wrong
        _choice("synthetic", "s-choice-1", "a"),
        _choice("synthetic", "s-choice-2", "b"),
        # score: 1 right
        _score("synthetic", "s-score-1", 0.0),
        # single-type source, all right
        _noul("sst2", "sst2-1", True),
    ]

    stats = eval_epoch_detailed(model, items)

    by_st = stats["by_source_type"]
    assert set(by_st) == {"synthetic", "sst2"}
    assert set(by_st["synthetic"]) == {"noul", "choice", "score"}

    noul = by_st["synthetic"]["noul"]
    assert noul["n_items"] == 3
    assert noul["correct"] == 2
    assert noul["accuracy"] == pytest.approx(2 / 3)

    choice = by_st["synthetic"]["choice"]
    assert choice["n_items"] == 2
    assert choice["accuracy"] == pytest.approx(0.5)

    score = by_st["synthetic"]["score"]
    assert score["n_items"] == 1
    assert score["accuracy"] == pytest.approx(1.0)

    assert by_st["sst2"]["noul"]["accuracy"] == pytest.approx(1.0)


def test_by_source_stays_a_sum_over_types():
    """Backward compatibility: by_source totals equal the per-type totals."""
    model = _ConstantModel(noul_logit=10.0, choice_best=0, score_best=0)
    items = [
        _noul("synthetic", "n1", True),
        _choice("synthetic", "c1", "b"),
        _score("synthetic", "v1", 1.0),
    ]

    stats = eval_epoch_detailed(model, items)

    merged = stats["by_source"]["synthetic"]
    per_type = stats["by_source_type"]["synthetic"]
    assert merged["n_items"] == sum(m["n_items"] for m in per_type.values())
    assert merged["correct"] == sum(m["correct"] for m in per_type.values())
    assert merged["correct"] == 1  # only noul is right


def test_aggregate_matches_by_type_totals():
    model = _ConstantModel(noul_logit=10.0, choice_best=0, score_best=0)
    items = [
        _noul("sst2", "n1", True),
        _noul("sst2", "n2", False),
        _choice("arc", "c1", "a"),
    ]

    stats = eval_epoch_detailed(model, items)

    assert stats["aggregate"]["n_items"] == 3
    assert stats["aggregate"]["correct"] == 2
    for t, m in stats["by_type"].items():
        rows = [
            mm
            for types in stats["by_source_type"].values()
            for tt, mm in types.items()
            if tt == t
        ]
        assert m["n_items"] == sum(r["n_items"] for r in rows)
        assert m["correct"] == sum(r["correct"] for r in rows)
