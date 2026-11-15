"""Torch-free helpers for the judgement plan rows.

Kept out of ``model/scripts/build_judgement_plan.py`` so the source, family,
cap and permutation rules are importable and testable without the model stack.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from collections import defaultdict

SYNTHETIC_TEMPLATE = re.compile(r"^synthetic-(.+)-(\d{4})-")


def source_of(item, fallback: str) -> str:
    """The item's own source; ``corpus`` is a mode name, not a source."""
    value = getattr(item, "source", None)
    return value if isinstance(value, str) and value else fallback


def synthetic_family(item) -> str | None:
    """The family embedded in a synthetic group key (``synthetic-<family>-NNNN``)."""
    if source_of(item, "") != "synthetic":
        return None
    group = getattr(item, "group", "") or ""
    parts = group.split("-")
    if len(parts) > 1 and parts[0] == "synthetic":
        return parts[1]
    return None


def synthetic_template(item) -> tuple[str, str] | None:
    """The generated item behind a synthetic row: ``(family, 4-digit item id)``."""
    if source_of(item, "") != "synthetic":
        return None
    match = SYNTHETIC_TEMPLATE.match(str(getattr(item, "id", "")))
    return (match.group(1), match.group(2)) if match else None


def namespace_of(item) -> str:
    """Split namespace: source, or source+family for synthetic rows."""
    source = source_of(item, "")
    family = synthetic_family(item)
    return f"synthetic/{family}" if source == "synthetic" and family else source


def permute_options(prompts: list, row_id: str, gold_slot: int | None = None) -> list:
    """Reorder a row's option prompts with a seed derived from ``row_id``.

    The gold option moves with the permutation (``option_index`` follows the
    new position), and every other field is untouched, so the option text
    multiset is unchanged.  When ``gold_slot`` is given, the single positive
    option is placed in that slot and the rest are shuffled into the remaining
    slots: the caller balances ``gold_slot`` across a stratum, which makes the
    gold position uniform instead of merely random.  Without this, the corpus
    order decides the gold position: choice K5 puts 52% of gold at index 0.
    """
    if len(prompts) <= 1:
        return prompts
    seed = int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:8], "big")
    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    positives = [index for index, prompt in enumerate(prompts) if prompt.label == 1]
    if gold_slot is not None and len(positives) == 1 and gold_slot < len(prompts):
        gold = positives[0]
        others = [index for index in order if index != gold]
        arranged = [None] * len(prompts)
        arranged[gold_slot] = prompts[gold]
        slots = [index for index in range(len(prompts)) if index != gold_slot]
        for slot, old_index in zip(slots, others):
            arranged[slot] = prompts[old_index]
        return [
            dataclasses.replace(prompt, option_index=index)
            for index, prompt in enumerate(arranged)
        ]
    return [
        dataclasses.replace(prompts[old_index], option_index=new_index)
        for new_index, old_index in enumerate(order)
    ]


def cap_per_template(records, cap: int):
    """Keep at most ``cap`` rows per synthetic template, round-robin over q_type.

    The kept rows are the lexicographically smallest row ids per q_type, so the
    result is deterministic and keeps every question type the template has.
    """
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    keep = []
    for record in records:
        template = synthetic_template(record[1])
        if template is None:
            keep.append(record)
        else:
            groups[template][record[3]].append(record)
    for by_type in groups.values():
        for q_type in by_type:
            by_type[q_type].sort(key=lambda record: record[0])
        picked = []
        index = 0
        while len(picked) < cap:
            added = False
            for q_type in sorted(by_type):
                if index < len(by_type[q_type]):
                    picked.append(by_type[q_type][index])
                    added = True
                    if len(picked) == cap:
                        break
            if not added:
                break
            index += 1
        keep.extend(picked)
    kept = {record[0] for record in keep}
    return [record for record in records if record[0] in kept]
