"""Tests for the CLINC150 item builder (#356).

The blocking and judgement cells all draw on this loader, and the 50-item
judgement subsample is a prefix of the 200-item cell. The builder is split from
the download so these properties are testable offline: the class-count guard,
stable ids, the subset-prefix property, and the exclusion of `oos`.
"""

from __future__ import annotations

import pytest

from model.evaluation import clinc


def names(n_in_scope: int = 150) -> list[str]:
    return [f"intent_{index:03d}" for index in range(n_in_scope)] + ["oos"]


def rows(n_rows: int = 300, n_in_scope: int = 150, oos_every: int = 50) -> list:
    """(row_index, text, intent_index) tuples; every ``oos_every``-th is oos."""
    out = []
    for index in range(n_rows):
        if index % oos_every == 0:
            intent = n_in_scope  # the oos slot
        else:
            intent = index % n_in_scope
        out.append((index, f"message number {index}", intent))
    return out


def test_build_rejects_a_non_150_class_space():
    with pytest.raises(SystemExit):
        clinc.build(rows(), names(149), 10)
    with pytest.raises(SystemExit):
        clinc.build(rows(), names(151), 10)


def test_build_keeps_150_classes_and_the_license_in_the_meta():
    items, meta = clinc.build(rows(), names(), 10)
    assert len(items) == 10
    assert meta["classes_in_scope"] == 150
    assert meta["revision"] == "155b9c710419136e17307b80d0a13e68cd46b4ec"
    assert meta["license"] == "CC BY 3.0"
    assert meta["repo"] == "clinc/clinc_oos"
    assert "first 10" in meta["item_selection"]
    assert len(items[0].question["criteria"]) == 150
    for item in items:
        assert item.label in item.question["criteria"]


def test_build_ids_are_stable_and_name_the_dataset_row():
    first, _ = clinc.build(rows(), names(), 20, seed=342)
    second, _ = clinc.build(rows(), names(), 20, seed=342)
    assert [item.id for item in first] == [item.id for item in second]
    # The id suffix is the dataset row index, so the same row keeps its id.
    for item in first:
        index = int(item.id.rsplit("-", 1)[-1])
        assert item.state == f"message number {index}"


def test_the_first_fifty_of_two_hundred_are_the_judgement_items():
    fifty, _ = clinc.build(rows(), names(), 50)
    two_hundred, _ = clinc.build(rows(), names(), 200)
    assert [item.id for item in fifty] == [item.id for item in two_hundred][:50]
    assert [item.state for item in fifty] == [item.state for item in two_hundred][:50]


def test_build_excludes_the_out_of_scope_rows():
    items, _ = clinc.build(rows(oos_every=50), names(), 300)
    for item in items:
        assert item.label != "oos"
        index = int(item.id.rsplit("-", 1)[-1])
        assert index % 50 != 0
