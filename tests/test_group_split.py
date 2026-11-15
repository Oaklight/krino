"""Group-aware split: cluster mates never land on both sides."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from model.training.group_split import (
    cluster_for,
    load_cluster_manifest,
    normalise_state,
    split_items_by_group,
    state_key,
)


def _item(item_id: str, state: str, group: str | None = None):
    return SimpleNamespace(id=item_id, state=state, group=group)


def _split(items, val_fraction=0.5, seed=0, **kwargs):
    return split_items_by_group(
        items,
        val_fraction,
        seed,
        id_of=lambda item: item.id,
        state_of=lambda item: item.state,
        group_of=lambda item: item.group,
        **kwargs,
    )


def _four():
    # p1 and p2 differ only by whitespace: one near-duplicate pair, plus two
    # unrelated rows.
    return [
        _item("p1", "Same  state"),
        _item("p2", "Same state"),
        _item("a", "alpha"),
        _item("b", "beta"),
    ]


@pytest.mark.feature_guard("KRINO_DISABLE_GROUP_SPLIT")
def test_exact_normalised_state_keeps_a_near_duplicate_pair_together():
    train, val, report = _split(_four())
    train_ids = {item.id for item in train}
    val_ids = {item.id for item in val}
    assert ("p1" in train_ids) == ("p2" in train_ids)
    assert ("p1" in val_ids) == ("p2" in val_ids)
    assert report["groups"] == 3  # pair + two singletons
    assert report["items"] == 4
    assert report["groups_with_over_one_item"] == 1


def test_cluster_manifest_merges_rows_with_different_states(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "state_to_cluster": {
                    state_key("alpha"): "nd-1",
                    state_key("beta"): "nd-1",
                }
            }
        )
    )
    clusters = load_cluster_manifest(path)
    items = [_item("p1", "alpha"), _item("p2", "beta"), _item("a", "gamma")]
    train, val, report = _split(
        items, cluster_of=lambda item: cluster_for(clusters, item.id, item.state)
    )
    train_ids = {item.id for item in train}
    assert ("p1" in train_ids) == ("p2" in train_ids)
    assert report["groups"] == 2
    assert report["clusters"] == 1


def test_id_mapping_manifest_shape_is_accepted(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"p1": "c1", "p2": "c1"}))
    assert load_cluster_manifest(path) == {"id:p1": "c1", "id:p2": "c1"}


def test_existing_group_key_keeps_a_passage_together():
    items = [
        _item("q1", "passage text part one", group="passage-1"),
        _item("q2", "a different question on the same passage", group="passage-1"),
        _item("a", "alpha", group="passage-2"),
    ]
    train, val, report = _split(items)
    train_ids = {item.id for item in train}
    val_ids = {item.id for item in val}
    assert ("q1" in train_ids) == ("q2" in train_ids)
    assert ("q1" in val_ids) == ("q2" in val_ids)
    assert report["groups"] == 2


def test_normalise_state_uses_the_235_key():
    assert normalise_state("  a\n\t b  ") == "a b"
    # NFC + casefold, the #235 normalization, so case-only variants group.
    assert normalise_state("Case") == normalise_state("case")


def test_zero_fraction_puts_everything_in_train():
    train, val, report = _split(_four(), val_fraction=0.0)
    assert len(train) == 4
    assert val == []
    assert report["val"] == 0


def test_full_fraction_puts_everything_in_val():
    train, val, report = _split(_four(), val_fraction=1.0)
    assert train == []
    assert len(val) == 4
    assert report["val"] == 4
