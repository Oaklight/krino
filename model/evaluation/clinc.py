"""CLINC150 items as a second, larger intent space (#356).

Banking77 has 80 intent texts and saturates at K=80. CLINC150 has 150 in-scope
classes, which is what makes K above the 84-code alphabet constructible.

The dataset is CC BY 3.0 and is pinned to a revision. It is never written into
the repository: callers record the id and revision and report results.

This module imports no torch, so both the calibration script and the paired
tier experiment can build the same items from one definition.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

REPO = "clinc/clinc_oos"
CONFIG = "plus"
SPLIT = "test"
REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
LICENSE = "CC BY 3.0"
INSTRUCTIONS = "Which intent category does this customer message belong to?"
SEED = 342


def build(
    rows: list[tuple[int, str, int]],
    names: list[str],
    count: int,
    seed: int = SEED,
) -> tuple[list[Any], dict[str, Any]]:
    """Items from the dataset's rows, without the download.

    ``rows`` holds ``(row_index, text, intent_index)`` in the dataset's own
    order. The row index is the item id suffix, so an id names the same CLINC
    row however many items the caller keeps, and the first ``count`` of one
    call are the first ``count`` of any larger call.

    The 150-class guard lives here so a loader change that alters the label
    space fails loudly instead of silently shifting every class index.
    """
    in_scope = [name for name in names if name != "oos"]
    if len(in_scope) != 150:
        raise SystemExit(f"expected 150 in-scope classes, got {len(in_scope)}")
    criteria = {name: name.replace("_", " ") for name in in_scope}
    kept = [
        (index, text, intent)
        for index, text, intent in rows
        if names[intent] != "oos"
    ]
    random.Random(seed).shuffle(kept)
    items = [
        SimpleNamespace(
            id=f"clinc150-{SPLIT}-{index:05d}",
            state=text,
            label=names[intent],
            question={"criteria": criteria, "instructions": INSTRUCTIONS},
        )
        for index, text, intent in kept[:count]
    ]
    meta = {
        "dataset": "clinc150",
        "repo": REPO,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "license": LICENSE,
        "classes_in_scope": len(in_scope),
        "instructions": INSTRUCTIONS,
        "item_selection": f"in-scope rows shuffled with Random({seed}), first {count}",
    }
    return items, meta


def load(count: int, seed: int = SEED) -> tuple[list[Any], dict[str, Any]]:
    """In-scope CLINC150 test items over one 150-class criteria dict.

    Rows are shuffled with a fixed seed and the first ``count`` are taken, so a
    subset is stable and independent of how many items the caller asks for.
    """
    from datasets import load_dataset

    data = load_dataset(REPO, CONFIG, split=SPLIT, revision=REVISION)
    names = data.features["intent"].names
    rows = [
        (index, row["text"], int(row["intent"])) for index, row in enumerate(data)
    ]
    return build(rows, names, count, seed)