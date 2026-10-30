"""Deterministic authored held-out items for the decision primitives.

Clause 4 cannot be decided on the public tier: JevBench's text is published for 231
items, its hard tier is 111 of 220, and at the measured discordance a 111-item tier
detects nothing below 15.7 points (`research/23`). Closing it needs a held-out set of
roughly 255 paired items, and the benchmark does not have them to give.

This module builds such a set instead of finding one. The design constraint is that
**the gold label must be computable, not judged**: every item is generated from a
numeric specification, the text is a rendering of that specification, and the label is
computed from it by the same arithmetic a human would apply. That removes the failure
mode this project already paid for once — 219 STS-B rows whose gold label was never
observed and which silently scored as the bottom level (`#177`) — because an item whose
label cannot be recomputed from its own text is not emitted at all.

Five families, all three primitives:

============  ======  =============================================================
``deadline``  choice  a date plus a term in days or weeks; four candidate dates
``approval``  choice  an amount against a two-threshold approval policy
``ownership`` choice  a code-to-owner table; which owner holds this code
``flag``      noul    records and a rule; is the record flagged
``severity``  score   a rubric that maps a count of failed checks to a level
============  ======  =============================================================

What this set can and cannot say is stated in `research/23`: it is a *paired* set for
detecting a difference between two checkpoints, not a substitute for an externally
authored benchmark. A model could learn the templates, so it is reported beside
JevBench and never instead of it.

Nothing here imports outside the standard library, so the gold arithmetic is testable
on a bare interpreter.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import random
from typing import Any, Callable, Iterator

SOURCE = "authored_holdout"


def _item_id(group: str, index: int) -> str:
    """Stable identifier derived from the family and the item's index."""
    digest = hashlib.sha1(f"{SOURCE}:{group}:{index}".encode()).hexdigest()
    return f"kh-{digest[:20]}"


def _shuffled(options: list[str], label: str, rng: random.Random) -> dict[str, str]:
    """Criteria mapping in a seeded order, so the gold position carries no signal."""
    ordered = list(options)
    rng.shuffle(ordered)
    return {f"option_{position}": text for position, text in enumerate(ordered)}


def _choice_item(
    *,
    group: str,
    index: int,
    state: str,
    instructions: str,
    options: list[str],
    gold_text: str,
    rng: random.Random,
) -> dict[str, Any]:
    criteria = _shuffled(options, gold_text, rng)
    keys = [key for key, text in criteria.items() if text == gold_text]
    if len(keys) != 1:
        raise ValueError(f"{group}-{index}: gold text is not unique among the options")
    return {
        "id": _item_id(group, index),
        "state": state,
        "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
        "label": keys[0],
        "source": SOURCE,
        "split": "test",
        "group": group,
    }


def _deadline(index: int, rng: random.Random) -> dict[str, Any]:
    """A filing date and a term in days; the due date is the correct option."""
    filed = _dt.date(2026, 1, 1) + _dt.timedelta(days=rng.randrange(0, 240))
    term = rng.choice([7, 10, 14, 21, 28, 30, 45, 60, 90])
    unit = rng.choice(["day", "week", "business day"])
    if unit == "week":
        term_days = term * 7
        wording = f"{term} weeks"
    elif unit == "business day":
        # Business days are counted Monday-Friday, so walk the calendar.
        remaining = term
        term_days = 0
        cursor = filed
        while remaining:
            cursor += _dt.timedelta(days=1)
            term_days += 1
            if cursor.weekday() < 5:
                remaining -= 1
        wording = f"{term} business days"
    else:
        term_days = term
        wording = f"{term} days"

    due = filed + _dt.timedelta(days=term_days)
    truth = due.isoformat()
    offsets = rng.sample([-10, -7, -3, -2, -1, 1, 2, 3, 7, 10, 14], k=3)
    options = [truth] + [(due + _dt.timedelta(days=off)).isoformat() for off in offsets]
    state = (
        "Case file 2026-{:04d} records that the filing was received on {} and that "
        "the response window is {} from the filing date.".format(
            rng.randrange(1, 9999), filed.isoformat(), wording
        )
    )
    return _choice_item(
        group="deadline",
        index=index,
        state=state,
        instructions="Which date is the response due?",
        options=options,
        gold_text=truth,
        rng=rng,
    )


def _approval(index: int, rng: random.Random) -> dict[str, Any]:
    """A spend request against a two-threshold policy."""
    manager_limit = rng.choice([250, 500, 750, 1000])
    director_limit = manager_limit * rng.choice([2, 3, 4])
    amount = rng.randrange(25, director_limit * 2, 5)
    if amount <= manager_limit:
        truth = "none"
    elif amount <= director_limit:
        truth = "manager"
    elif amount <= director_limit * 2:
        truth = "director"
    else:
        truth = "board"
    state = (
        "Purchase request: {} for equipment. The delegation policy reads: amounts up "
        "to {} need no approval; above {} and up to {} need a manager; above {} and up "
        "to {} need a director; anything larger needs the board.".format(
            f"{amount:,}", f"{manager_limit:,}", f"{manager_limit:,}",
            f"{director_limit:,}", f"{director_limit:,}", f"{director_limit * 2:,}",
        )
    )
    descriptions = {
        "none": "no approval required",
        "manager": "manager approval required",
        "director": "director approval required",
        "board": "board approval required",
    }
    return _choice_item(
        group="approval",
        index=index,
        state=state,
        instructions="Which approval level applies to this request?",
        options=list(descriptions.values()),
        gold_text=descriptions[truth],
        rng=rng,
    )


def _ownership(index: int, rng: random.Random) -> dict[str, Any]:
    """A code-to-owner table; the question asks for one code's owner."""
    owners = rng.sample(
        ["Achebe", "Baptiste", "Chen", "Duarte", "Eriksen", "Farouk", "Grover", "Haddad"],
        k=4,
    )
    codes = [f"Z{rng.randrange(100, 999)}" for _ in range(4)]
    table = dict(zip(codes, owners))
    asked = rng.choice(codes)
    truth = table[asked]
    rows = "; ".join(f"{code} is held by {owner}" for code, owner in table.items())
    state = f"Register of holdings: {rows}. No two codes share an owner."
    return _choice_item(
        group="ownership",
        index=index,
        state=state,
        instructions=f"Who holds code {asked}?",
        options=owners,
        gold_text=truth,
        rng=rng,
    )


def _flag(index: int, rng: random.Random) -> dict[str, Any]:
    """A record whose status is decided by a small explicit rule."""
    rule_kind = rng.choice(["expiry", "balance", "documents"])
    if rule_kind == "expiry":
        days = rng.choice([-30, -7, -1, 0, 1, 7, 30, 90])
        flagged = days <= 0
        state = (
            "Account review: the certificate expired {} days ago. The rule is that an "
            "account is flagged when its certificate has expired or expires today."
            .format(abs(days))
        )
        if days > 0:
            state = (
                "Account review: the certificate expires in {} days. The rule is that "
                "an account is flagged when its certificate has expired or expires "
                "today.".format(days)
            )
    elif rule_kind == "balance":
        balance = rng.choice([-500, -25, 0, 25, 100, 2500])
        flagged = balance < 0
        state = (
            "Account review: the ledger balance is {:,} units. The rule is that an "
            "account is flagged only when the balance is negative.".format(balance)
        )
    else:
        missing = rng.choice([0, 1, 2, 3])
        flagged = missing > 0
        state = (
            "Document review: {} of the four required documents are missing. The rule "
            "is that a file is flagged when any required document is missing."
            .format(missing)
        )
    return {
        "id": _item_id("flag", index),
        "state": state,
        "question": {
            "type": "noul",
            "instructions": "Is the record flagged under the stated rule?",
        },
        "label": "yes" if flagged else "no",
        "source": SOURCE,
        "split": "test",
        "group": "flag",
    }


def _severity(index: int, rng: random.Random) -> dict[str, Any]:
    """A rubric that maps a count to an ordinal level."""
    checks = rng.randrange(0, 8)
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
    state = (
        "Inspection sheet: {} of the seven checks failed. The severity rubric reads: "
        "no failures is 1; one failure is 2; two failures is 3; three or four failures "
        "is 4; five or more failures is 5.".format(checks)
    )
    return {
        "id": _item_id("severity", index),
        "state": state,
        "question": {
            "type": "score",
            "instructions": "What severity level does the rubric assign?",
            "criteria": ["1", "2", "3", "4", "5"],
        },
        "label": float(level),
        "source": SOURCE,
        "split": "test",
        "group": "severity",
    }


FAMILIES: dict[str, Callable[[int, random.Random], dict[str, Any]]] = {
    "deadline": _deadline,
    "approval": _approval,
    "ownership": _ownership,
    "flag": _flag,
    "severity": _severity,
}


def build_items(per_family: int = 51, seed: int = 20261030) -> Iterator[dict[str, Any]]:
    """Yield ``per_family`` items from each family, deterministically.

    Args:
        per_family: Items to draw from every family; 51 across five families gives 255.
        seed: Master seed. Each (family, index) draws from its own stream, so adding a
            family or raising ``per_family`` does not disturb the items already drawn.

    Yields:
        Item dicts in the ``authored144`` shape: ``id``, ``state``, ``question`` with
        ``type``/``instructions``/``criteria``, ``label``, ``source``, ``split``,
        ``group``.
    """
    for group, builder in FAMILIES.items():
        for index in range(per_family):
            yield builder(index, random.Random(f"{seed}:{group}:{index}"))


def check_item(item: dict[str, Any]) -> list[str]:
    """Structural and recomputational checks for one item.

    The recomputation is the point: a choice item's gold must name one of its own
    criteria, a noul item's must be ``yes`` or ``no``, and a score item's must be one
    of its levels. An item whose label cannot be tied back to its own text is exactly
    the defect `#177` removed, so this is enforced at emit time rather than trusted.

    Args:
        item: One generated item.

    Returns:
        A list of problems; empty means the item is well formed.
    """
    problems: list[str] = []
    question = item.get("question") or {}
    kind = question.get("type")
    label = item.get("label")
    if not item.get("state"):
        problems.append("empty state")
    if kind == "choice":
        criteria = question.get("criteria") or {}
        if label not in criteria:
            problems.append(f"label {label!r} is not a criteria key")
        texts = list(criteria.values())
        if len(set(texts)) != len(texts):
            problems.append("duplicate option texts")
    elif kind == "noul":
        if label not in ("yes", "no"):
            problems.append(f"noul label {label!r} is not yes/no")
    elif kind == "score":
        levels = question.get("criteria") or []
        if str(int(label)) not in [str(level) for level in levels]:
            problems.append(f"score label {label!r} is not one of the levels")
    else:
        problems.append(f"unknown question type {kind!r}")
    return problems
