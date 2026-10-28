"""Tests for the data-summary line the training script prints.

The line used to report `question["type"]`, which score-form conversion
rewrites to "score" for every item.  A converted run therefore printed
`{'score': 182262}` even though the sampler was still pooling the original
three types — a banner that reads as a collapsed mix when nothing collapsed.
"""

import pytest

torch = pytest.importorskip("torch")

from data.format import TypedQuestion
from data.score_form import convert_training_items
from model.scripts.train_multitask import print_data_summary


def _item(q_type: str, label, idx: int, source: str = "src") -> TypedQuestion:
    if q_type == "noul":
        question = {"type": "noul", "instructions": "is it so?"}
    elif q_type == "choice":
        question = {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {"a": "first", "b": "second"},
        }
    else:
        question = {
            "type": "score",
            "instructions": "rate it",
            "criteria": ["low", "mid", "high"],
        }
    return TypedQuestion(
        id=f"{q_type}-{idx}",
        state="a state",
        question=question,
        label=label,
        source=source,
        split="train",
    )


def _mixed(n: int = 3) -> list[TypedQuestion]:
    items: list[TypedQuestion] = []
    for i in range(n):
        items.append(_item("noul", True, i))
        items.append(_item("choice", "a", i))
        items.append(_item("score", 1, i))
    return items


class TestTypeDistribution:
    def test_unconverted_items_report_declared_types(self, capsys):
        print_data_summary(_mixed(), [])
        out = capsys.readouterr().out
        assert "'noul': 3" in out
        assert "'choice': 3" in out
        assert "'score': 3" in out
        assert "items are in score form" not in out

    def test_converted_items_report_original_types(self, capsys):
        converted = convert_training_items(_mixed(), "single")
        print_data_summary(converted, [])
        out = capsys.readouterr().out

        # Every item now declares type "score"; the summary must still show
        # the mix the sampler will actually draw from.
        assert "'noul': 3" in out
        assert "'choice': 3" in out
        assert "'score': 3" in out

    def test_converted_run_says_why(self, capsys):
        converted = convert_training_items(_mixed(), "single")
        print_data_summary(converted, [])
        out = capsys.readouterr().out
        # Only noul and choice are rewritten; the three score items were
        # already in score form and carry no score_form metadata.
        assert "6 items are in score form" in out
        assert "type_ratios still apply" in out

    def test_already_score_items_are_not_counted_as_converted(self, capsys):
        converted = convert_training_items(_mixed(), "single")
        marked = sum(
            1 for item in converted
            if isinstance(item.question.get("score_form"), dict)
        )
        assert marked == 6
        assert len(converted) == 9

    def test_mixed_run_counts_only_converted(self, capsys):
        items = _mixed(2)
        converted = convert_training_items(items, "single")
        print_data_summary(items + converted, [])
        out = capsys.readouterr().out
        assert "4 items are in score form" in out
        # 6 plain + 6 converted, pooled by original type.
        assert "'noul': 4" in out
        assert "'choice': 4" in out
        assert "'score': 4" in out

    def test_candidates_expansion_keeps_the_mix(self, capsys):
        converted = convert_training_items(_mixed(), "candidates")
        print_data_summary(converted, [])
        out = capsys.readouterr().out
        # choice questions become one score question per option, so the
        # choice count doubles; noul and score are unchanged.
        assert "'noul': 3" in out
        assert "'choice': 6" in out
        assert "'score': 3" in out
