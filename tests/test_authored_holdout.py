"""Tests for the deterministic authored held-out generator.

The point of these tests is not that the generator runs — it is that every gold label
can be re-derived from the emitted text by a parser that does not share the generator's
code path. That is the property `#177` showed a benchmark can lack: labels that do not
follow from the item they are attached to.
"""

from __future__ import annotations

import collections
import datetime as dt
import re

import pytest

from data.authored_holdout import FAMILIES, build_items, check_item


@pytest.fixture(scope="module")
def items():
    return list(build_items(per_family=51, seed=20261030))


def _by_group(items, group):
    return [item for item in items if item["group"] == group]


def _gold_text(item):
    """The option text the label points at, for a choice item."""
    return item["question"]["criteria"][item["label"]]


def test_build_is_deterministic():
    first = list(build_items(per_family=5, seed=7))
    second = list(build_items(per_family=5, seed=7))
    assert first == second


def test_a_different_seed_changes_the_items():
    first = list(build_items(per_family=5, seed=7))
    second = list(build_items(per_family=5, seed=8))
    assert [item["state"] for item in first] != [item["state"] for item in second]


def test_counts_and_primitive_mix(items):
    assert len(items) == 255
    kinds = collections.Counter(item["question"]["type"] for item in items)
    assert kinds == {"choice": 153, "noul": 51, "score": 51}
    assert set(item["group"] for item in items) == set(FAMILIES)


def test_every_item_passes_its_own_structural_checks(items):
    problems = [(item["id"], issue) for item in items for issue in check_item(item)]
    assert problems == []


def test_identifiers_are_unique(items):
    ids = [item["id"] for item in items]
    assert len(set(ids)) == len(ids)


def test_gold_option_position_carries_no_signal(items):
    choice = [item for item in items if item["question"]["type"] == "choice"]
    positions = collections.Counter(item["label"] for item in choice)
    assert set(positions) == {f"option_{index}" for index in range(4)}
    # No position may hold more than 40% of the golds in a balanced 4-way set.
    assert max(positions.values()) / len(choice) < 0.40


def test_deadline_gold_follows_from_the_stated_date_and_term(items):
    for item in _by_group(items, "deadline"):
        match = re.search(r"received on (\d{4}-\d{2}-\d{2})", item["state"])
        term = re.search(r"window is (\d+) (day|week|business day)s?", item["state"])
        assert match and term, item["state"]
        filed = dt.date.fromisoformat(match.group(1))
        count, unit = int(term.group(1)), term.group(2)
        if unit == "week":
            due = filed + dt.timedelta(weeks=count)
        elif unit == "day":
            due = filed + dt.timedelta(days=count)
        else:
            cursor, remaining, walked = filed, count, 0
            while remaining:
                cursor += dt.timedelta(days=1)
                walked += 1
                if cursor.weekday() < 5:
                    remaining -= 1
            due = filed + dt.timedelta(days=walked)
        assert _gold_text(item) == due.isoformat(), item["state"]


def test_approval_gold_follows_from_the_stated_policy(items):
    for item in _by_group(items, "approval"):
        amount = int(re.search(r"request: ([\d,]+) for equipment", item["state"]).group(1).replace(",", ""))
        no_approval = int(re.search(r"up to ([\d,]+) need no approval", item["state"]).group(1).replace(",", ""))
        manager = int(re.search(r"and up to ([\d,]+) need a manager", item["state"]).group(1).replace(",", ""))
        director = int(re.search(r"and up to ([\d,]+) need a director", item["state"]).group(1).replace(",", ""))
        if amount <= no_approval:
            expected = "no approval required"
        elif amount <= manager:
            expected = "manager approval required"
        elif amount <= director:
            expected = "director approval required"
        else:
            expected = "board approval required"
        assert _gold_text(item) == expected, item["state"]
        assert item["label"] in item["question"]["criteria"]


def test_ownership_gold_follows_from_the_stated_table(items):
    for item in _by_group(items, "ownership"):
        table = dict(re.findall(r"(Z\d+) is held by (\w+)", item["state"]))
        asked = re.search(r"code (Z\d+)", item["question"]["instructions"]).group(1)
        assert _gold_text(item) == table[asked], item["state"]


def test_flag_gold_follows_from_the_stated_rule(items):
    for item in _by_group(items, "flag"):
        state = item["state"]
        expired = re.search(r"expired (\d+) days ago", state)
        expiring = re.search(r"expires in (\d+) days", state)
        balance = re.search(r"balance is (-?[\d,]+) units", state)
        missing = re.search(r"(\d+) of the four required documents are missing", state)
        if expired or expiring:
            flagged = bool(expired)  # expired (including today) is flagged
        elif balance:
            flagged = int(balance.group(1).replace(",", "")) < 0
        elif missing:
            flagged = int(missing.group(1)) > 0
        else:  # pragma: no cover - guards against a new rendering path
            raise AssertionError(state)
        assert item["label"] == ("yes" if flagged else "no"), state


def test_flag_labelled_instances_exist_on_both_sides(items):
    labels = collections.Counter(item["label"] for item in _by_group(items, "flag"))
    # Balanced by construction: a lopsided family lets a constant answer score above
    # chance, which is exactly what the first version of this family allowed.
    assert labels["yes"] >= 20 and labels["no"] >= 20


def test_severity_gold_follows_from_the_stated_rubric(items):
    for item in _by_group(items, "severity"):
        checks = int(re.search(r"(\d+) of the seven checks failed", item["state"]).group(1))
        if checks == 0:
            level = 1
        elif checks == 1:
            level = 2
        elif checks == 2:
            level = 3
        elif checks <= 4:
            level = 4
        else:
            level = 5
        # Labels are level indices 0-4; the rubric in the text reads 1-5.
        assert item["label"] == float(level - 1), item["state"]
        assert item["question"]["criteria"] == ["1", "2", "3", "4", "5"]
        assert 0 <= item["label"] <= 4


def test_score_levels_are_all_exercised(items):
    levels = collections.Counter(item["label"] for item in _by_group(items, "severity"))
    assert set(levels) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert min(levels.values()) >= 3


def test_score_labels_are_level_indices_not_display_levels(items):
    """The project's score convention is a 0-based index; a 1-based label is dropped.

    `data.sources.invalid_score_labels` keeps only labels on `0 .. n_levels-1`, and the
    first version of this set emitted 1-5 and silently lost ten of its 51 score items.
    """
    from data.sources import invalid_score_labels

    score = [item for item in items if item["question"]["type"] == "score"]
    assert score and invalid_score_labels([]) == []

    class _Stub:
        def __init__(self, question, label):
            self.question = question
            self.label = label

    stubs = [_Stub(item["question"], item["label"]) for item in score]
    assert invalid_score_labels(stubs) == []


def test_growing_the_set_does_not_disturb_existing_items():
    small = list(build_items(per_family=3, seed=11))
    large = {item["id"]: item for item in build_items(per_family=6, seed=11)}
    for item in small:
        assert large[item["id"]] == item


def test_loader_reads_the_generated_file(tmp_path, monkeypatch):
    import json

    import data.pipeline as pipeline_mod

    target = tmp_path / "openjev" / "authored_holdout.jsonl"
    target.parent.mkdir(parents=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in build_items(per_family=2, seed=5):
            handle.write(json.dumps(item) + "\n")
    monkeypatch.setattr(pipeline_mod, "DATA_DIR", tmp_path)

    loaded = list(pipeline_mod.load_authored_holdout())
    assert len(loaded) == 10
    assert {question.question["type"] for question in loaded} == {
        "choice", "noul", "score"
    }
    assert all(question.split == "test" for question in loaded)
    score = [question for question in loaded if question.question["type"] == "score"]
    assert score and all(isinstance(question.label, float) for question in score)
    choice = [question for question in loaded if question.question["type"] == "choice"]
    assert all(question.label in question.question["criteria"] for question in choice)


def test_loader_falls_back_to_the_generator_when_the_file_is_absent(tmp_path, monkeypatch):
    import data.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "DATA_DIR", tmp_path)
    loaded = list(pipeline_mod.load_authored_holdout())
    assert len(loaded) == 255
    assert loaded[0].id == list(build_items())[0]["id"]


def test_committed_file_matches_the_generator():
    """The tracked JSONL must be exactly what the generator produces.

    The file is the artefact a model is scored on, so it is committed; the generator
    is how it is rebuilt. If the two drift, the benchmark and the code disagree about
    what was measured, which is the failure this test exists to prevent.
    """
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "data" / "benchmarks" / "openjev" / "authored_holdout.jsonl"
    )
    committed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = list(build_items(per_family=51, seed=20261030))
    assert committed == generated


def test_eval_config_scores_the_tracked_file_and_drops_nothing():
    """The eval config must resolve to the committed file, not a copy beside it.

    `load_source_items` does not dispatch through `LOADERS`; it reads whatever JSONL
    the source config names, defaulting to `data/benchmarks/<source>.jsonl`. The first
    scoring run named nothing, found a stale copy at that default path written before
    the score labels became level indices, and lost ten of the 51 score items to the
    `0..n_levels-1` check — with the run still reporting a clean score number. This
    test runs the config's own source entry and requires the whole set back.
    """
    import random
    from pathlib import Path

    import yaml

    from data.sources import load_source_items, resolve_path

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "model" / "configs" / "eval_authored.yaml").read_text(encoding="utf-8"))
    source_cfg = config["sources"]["authored_holdout"]
    assert resolve_path(source_cfg["path"]) == root / "data" / "benchmarks" / "openjev" / "authored_holdout.jsonl"

    train_items, eval_items = load_source_items("authored_holdout", source_cfg, random.Random(42))
    assert train_items == []  # weight 0: an eval-only set never trains
    kinds = collections.Counter(item.question["type"] for item in eval_items)
    assert kinds == {"choice": 153, "noul": 51, "score": 51}
