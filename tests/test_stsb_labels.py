"""The STS-B label sentinel must not become a gold label.

GLUE's STS-B test split ships `label = -1.0` on every row because the labels are
withheld. The loader used to clamp that into `0..5`, so 1,379 unlabeled rows
entered the eval set as gold level 0 — half the stsb eval slice. These tests pin
the sentinel handling, and the last one pins the loader's own behaviour so the
clamp cannot come back.
"""

from __future__ import annotations

import pytest

import data.pipeline as pipeline


@pytest.mark.parametrize("raw,expected", [(0.0, 0), (0.49, 0), (2.5, 2), (5.0, 5)])
def test_scores_map_to_levels(raw, expected):
    assert pipeline.stsb_level(raw) == expected


@pytest.mark.parametrize("raw", [-1.0, -0.5, -1])
def test_the_withheld_label_sentinel_is_not_a_level(raw):
    """-1 is GLUE's 'no label here', not 'no similarity'."""
    assert pipeline.stsb_level(raw) is None


def test_values_above_the_scale_are_clamped_not_dropped():
    assert pipeline.stsb_level(5.4) == 5
    assert pipeline.stsb_level(9.0) == 5


@pytest.mark.parametrize("raw", [None, "not a number", ""])
def test_missing_or_unparseable_labels_are_dropped(raw):
    assert pipeline.stsb_level(raw) is None


def test_the_loader_drops_withheld_rows_instead_of_calling_them_zero(monkeypatch):
    """A -1 row must not reach the eval set as level 0."""
    rows = {
        "train": [{"sentence1": "a", "sentence2": "b", "label": 3.0}],
        "validation": [{"sentence1": "c", "sentence2": "d", "label": 0.0}],
        "test": [
            {"sentence1": "e", "sentence2": "f", "label": -1.0},
            {"sentence1": "g", "sentence2": "h", "label": -1.0},
        ],
    }
    monkeypatch.setattr(
        pipeline, "_load_hf_parquet", lambda repo, config, split: rows[split]
    )
    items = list(pipeline.load_stsb())

    assert len(items) == 2, "the two unlabeled test rows must be skipped"
    assert [item.split for item in items] == ["train", "test"]
    assert all(float(item.label) >= 0 for item in items)
    # The one real zero in validation survives: the fix must not drop genuine
    # "No similarity" rows, only the sentinel.
    assert [float(item.label) for item in items] == [3.0, 0.0]
