"""The judgement readout's option construction (#277 follow-up).

The readout decides what text sits after ``Option:`` and which index is the gold
answer. Both are easy to get wrong in a way that still produces a number, so they
are pinned here without loading a model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from model.evaluation.judgement_prompt import (
    JUDGEMENT_TEMPLATE,
    NO,
    POLARITY_TEMPLATE,
    YES,
    candidates,
    label_polarity,
    polarity_gold,
    render,
    render_polarity,
)


def _item(q_type: str, criteria, label, instructions="Do the thing."):
    return SimpleNamespace(
        id="easy-intent-00",
        state="A situation.",
        question={"type": q_type, "instructions": instructions, "criteria": criteria},
        label=label,
    )


def test_noul_keeps_the_questions_own_wording_true_first():
    texts, gold = candidates(_item("noul", {"true": "Sí", "false": "No"}, True))
    assert texts == ["Sí", "No"]
    assert gold == 0


def test_noul_without_criteria_is_not_an_option_set():
    # No criteria means no option text to judge; the caller reads polarity at
    # Answer: instead of asking whether the word "yes" answers the question.
    assert candidates(_item("noul", None, False)) is None
    assert candidates(_item("noul", {}, True)) is None


def test_noul_string_label_picks_the_true_option():
    # JevBench stores expected as the strings "yes" and "no". Comparing against
    # "true" alone marked all 74 of its noul items negative (#299).
    criteria = {"true": "Yes, it holds.", "false": "No, it does not."}
    texts, gold = candidates(_item("noul", criteria, "yes"))
    assert texts == ["Yes, it holds.", "No, it does not."]
    assert gold == 0
    assert candidates(_item("noul", criteria, "no"))[1] == 1
    assert candidates(_item("noul", criteria, " YES "))[1] == 0
    assert candidates(_item("noul", criteria, "False"))[1] == 1
    assert candidates(_item("noul", criteria, 1))[1] == 0
    assert candidates(_item("noul", criteria, 0))[1] == 1


def test_noul_unknown_label_refuses_to_pick_a_gold_option():
    # A silent default to negative is how the JevBench labels were inverted; an
    # unrecognised form is a data error and must stop the run.
    criteria = {"true": "Yes.", "false": "No."}
    with pytest.raises(ValueError, match="is neither true nor false"):
        candidates(_item("noul", criteria, "maybe"))


def test_build_prompts_marks_the_true_option_for_a_string_yes_label():
    # The plan builder and the scoring path both go through build_prompts, so
    # the label has to survive the trip as a per-option 0/1 flag.
    pytest.importorskip("torch")
    from model.src.judgement_objective import build_prompts

    prompts = build_prompts(_item("noul", {"true": "Yes.", "false": "No."}, "yes"))
    assert [p.option_index for p in prompts] == [0, 1]
    assert [p.label for p in prompts] == [1, 0]


def test_label_polarity_names_the_forms_it_accepts():
    for value in (True, "true", "yes", "1", "Y", " Yes "):
        assert label_polarity(value) is True, value
    for value in (False, "false", "no", "0", "N", " No "):
        assert label_polarity(value) is False, value
    assert label_polarity(1) is True
    assert label_polarity(0) is False
    for value in ("maybe", None, 2, 0.5, ""):
        assert label_polarity(value) is None, value


def test_no_scorer_carries_its_own_noul_label_parse():
    # Milo's review on #378: six copies of the same rule in four spellings. The
    # scan covers the whole repository rather than one spelling in one tree, so
    # a seventh copy fails unless it is added to the allowlist on purpose.
    import re
    from pathlib import Path

    pattern = re.compile(r"""\.lower\(\)\s*==\s*["']true["']""")
    root = Path(__file__).resolve().parents[1]
    found: dict[str, list[str]] = {}
    for tree in ("model", "data", "scripts"):
        base = root / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            hits = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if pattern.search(line)
            ]
            if hits:
                found[path.relative_to(root).as_posix()] = hits

    # Corpus-side sites with different semantics, kept out of #378 on purpose
    # and tracked as follow-up on #299.
    allowed = {
        "model/scripts/build_unanswerable_pairs.py":
            "`label is True or ... == \"true\"` variant, corpus-side, reads bools",
        "model/scripts/train_residual.py":
            "training target rather than a scorer polarity, reads bools",
    }
    unexpected = {name: hits for name, hits in found.items() if name not in allowed}
    assert unexpected == {}, unexpected
    # An allowlist entry that no longer matches is stale and must be removed.
    assert set(found) == set(allowed), (sorted(found), sorted(allowed))


def test_a_corpus_noul_label_of_either_form_resolves_through_the_parser(tmp_path):
    # `load_jsonl` is TypedQuestion(**d) and coerces nothing, so the scorers
    # must not depend on the label arriving as a bool. Both on-disk forms go
    # through the shared parser to the same answer.
    from data.format import TypedQuestion
    from data.pipeline import load_jsonl, save_jsonl
    from model.evaluation.polarity import require_polarity

    def item(item_id, label):
        return TypedQuestion(
            id=item_id, state="A situation.", source="corpus", split="train",
            label=label,
            question={"type": "noul", "instructions": "Does it hold?",
                      "criteria": {"true": "Yes.", "false": "No."}},
        )

    path = tmp_path / "corpus.jsonl"
    save_jsonl([item("corpus:a", True), item("corpus:b", "yes")], path)
    loaded = load_jsonl(path)
    assert loaded[0].label is True
    assert loaded[1].label == "yes"
    assert [require_polarity(row.label, row.id) for row in loaded] == [True, True]
    with pytest.raises(ValueError, match="is neither true nor false"):
        require_polarity("maybe", "corpus:c")


def test_polarity_gold_reads_a_boolean_label():
    assert polarity_gold(_item("noul", None, True)) is True
    assert polarity_gold(_item("noul", None, False)) is False
    assert polarity_gold(_item("noul", None, "yes")) is True
    assert polarity_gold(_item("noul", None, "False")) is False


def test_polarity_branch_never_fires_on_a_criteria_bearing_noul():
    # Every JevBench noul item carries criteria, so the acceptance set cannot
    # reach this branch and the repair cannot be tuned against it.
    assert polarity_gold(_item("noul", {"true": "Sí", "false": "No"}, True)) is None


def test_polarity_branch_is_noul_only():
    assert polarity_gold(_item("choice", None, "a")) is None
    assert polarity_gold(_item("score", ["low", "high"], 1)) is None


def test_polarity_template_has_no_option_line():
    assert render_polarity("S", "Q") == "State: S\nQuestion: Q\nAnswer:"
    assert "Option:" not in POLARITY_TEMPLATE


def test_choice_presents_descriptions_and_indexes_the_gold_key():
    item = _item("choice", {"a": "First", "b": "Second", "c": "Third"}, "b")
    texts, gold = candidates(item)
    assert texts == ["First", "Second", "Third"]
    assert gold == 1


def test_choice_returns_none_when_the_label_is_not_an_option():
    assert candidates(_item("choice", {"a": "First"}, "zzz")) is None


def test_score_presents_named_levels_and_rounds_the_gold():
    texts, gold = candidates(_item("score", ["low", "mid", "high"], 2.4))
    assert texts == ["Level 0: low", "Level 1: mid", "Level 2: high"]
    assert gold == 2


def test_template_ends_with_the_answer_cue():
    assert render("S", "Q", "O") == (
        "State: S\nQuestion: Q\nOption: O\n"
        "Does this option correctly answer the question? Answer:"
    )
    assert JUDGEMENT_TEMPLATE.endswith("Answer:")


def test_the_scored_pair_is_yes_then_no():
    assert (YES, NO) == (" yes", " no")


def test_record_correct_honours_an_explicit_outcome():
    from model.evaluation.jevbench import record_correct

    # A one-proposition record states its own outcome; the argmax over a single
    # logit would otherwise be correct by construction.
    assert record_correct(
        {"logits": [1.0], "label_order": [0], "label_idx": 0, "correct": False}
    ) is False
    assert record_correct(
        {"logits": [-1.0], "label_order": [0], "label_idx": 0, "correct": True}
    ) is True


class _FixedScorer:
    """Return the same ``(log P(yes), log P(no))`` pair for every option."""

    def __init__(self, yes: float, no: float):
        self.yes = yes
        self.no = no
        self.prompts: list[str] = []

    def score_options(self, prompt, options):
        assert options == [" yes", " no"]
        self.prompts.append(prompt)
        return [self.yes, self.no]


def test_option_branch_ranks_on_yes_and_records_the_no_side():
    # #310: the ranking statistic is log P(" yes") alone. log P(" no") is
    # recorded beside it, and must not enter ``logits``.
    pytest.importorskip("torch")
    from model.scripts.eval_jevbench_judgement import prediction_row, score_item

    scorer = _FixedScorer(yes=-0.75, no=-2.25)
    item = _item("choice", {"a": "First", "b": "Second"}, "a")
    record = score_item(scorer, item)

    assert record["mode"] == "option"
    assert record["logits"] == [-0.75, -0.75]
    assert record["logp_yes"] == [-0.75, -0.75]
    assert record["logp_no"] == [-2.25, -2.25]
    assert record["label_idx"] == 0
    assert len(scorer.prompts) == 2

    row = prediction_row(item, record)
    assert row["logp_yes"] == [-0.75, -0.75]
    assert row["logp_no"] == [-2.25, -2.25]
    assert row["predicted_idx"] == 0
    assert row["mode"] == "option"


def test_polarity_branch_records_both_sides_and_keeps_the_difference():
    # A criteria-less noul item is thresholded on log P(" yes") - log P(" no"),
    # so ``logits`` stays the signed difference while both sides are recorded.
    pytest.importorskip("torch")
    from model.scripts.eval_jevbench_judgement import prediction_row, score_item

    scorer = _FixedScorer(yes=-0.5, no=-1.5)
    item = _item("noul", None, True)
    record = score_item(scorer, item)

    assert record["mode"] == "polarity"
    assert record["logits"] == [-0.5 - -1.5]
    assert record["logp_yes"] == [-0.5]
    assert record["logp_no"] == [-1.5]
    assert record["correct"] is True
    assert len(scorer.prompts) == 1

    # The persisted row must not put the difference under a log-probability
    # name; that mapping is a separate step from the one ``score_item`` does.
    row = prediction_row(item, record)
    assert row["logp_yes"] == [-0.5]
    assert row["logp_no"] == [-1.5]
    assert row["predicted_idx"] == 0
    assert row["mode"] == "polarity"
