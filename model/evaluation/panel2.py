"""Vocabulary-code readout (panel 2): prompt, options, and diagnostics.

One forward with the context and every option in a single sequence.  Each
option carries a one-token code, and the answer is the argmax of the next-token
distribution at the final position restricted to those code ids.  This module
holds the parts that decide what is shown and how the diagnostics are computed,
with no torch import, so the script that runs the model and the tests share one
definition.

The prompt is fixed in :data:`PANEL2_TEMPLATE` and built only by
:func:`render_panel2`, so all four checkpoints see the same text and the
comparison is a comparison of weights rather than of prompts (#326).

**Tie-breaking is part of the readout.** When two code logits are exactly
equal the answer must not depend on which option was displayed first, or an
order-sensitivity measurement counts its own convention as an effect. The
panel-2 rule is therefore: map each displayed code's logit onto its natural
option index, then take the first maximum in that index order
(:func:`choose_option`). Every harness that reads code logits goes through it,
and every run records how many items were tied (:func:`has_exact_tie`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

CODES = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!@#$%^&*()-_=+[]{}<>?|~"
    "\\/\"'`"
    ",.;:"
)

PANEL2_TEMPLATE = (
    "State: {state}\n"
    "Question: {instructions}\n"
    "Options:\n"
    "{options}\n"
    "Answer:"
)

POSITION_DOMINATED_PP = 10.0


def codes_for(n_options: int) -> list[str]:
    """The codes for an item with ``n_options`` options.

    A to Z for the first 26 options, then a to z, then single-character
    symbols.  Every code in the alphabet encodes to one token in the Qwen3
    tokenizers; the dev set's 80-option items are the reason the alphabet goes
    past 26.
    """
    if not 2 <= n_options <= len(CODES):
        raise ValueError(f"n_options must be in [2, {len(CODES)}], got {n_options}")
    return list(CODES[:n_options])


def option_lines(texts: Sequence[str], codes: Sequence[str]) -> str:
    """One line per option, as ``{code}. {option text}``."""
    if len(texts) != len(codes):
        raise ValueError(f"{len(texts)} option texts but {len(codes)} codes")
    return "\n".join(f"{code}. {text}" for code, text in zip(codes, texts))


def render_panel2(
    state: str, instructions: str, texts: Sequence[str], codes: Sequence[str]
) -> str:
    """The single prompt form used by every checkpoint."""
    return PANEL2_TEMPLATE.format(
        state=state,
        instructions=instructions,
        options=option_lines(texts, codes),
    )


def options_for(item: Any) -> tuple[list[str], int] | None:
    """The option texts shown to panel 2, and the gold index.

    A noul item is rendered as the same two-option question as every other
    item -- ``A. no`` and ``B. yes`` -- so the 231-item total stays comparable
    to the judgement baseline.  Its own criteria wording is not used.
    """
    question = item.question
    q_type = question.get("type", "noul")

    if q_type == "noul":
        label = item.label
        if isinstance(label, bool):
            positive = label
        else:
            positive = str(label).strip().lower() in ("true", "yes", "1", "y")
        return ["no", "yes"], 1 if positive else 0

    if q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria)
        if item.label not in criteria:
            return None
        return [str(criteria[key] or key) for key in keys], keys.index(item.label)

    if q_type == "score":
        criteria = question["criteria"]
        texts = [f"Level {index}: {desc}" for index, desc in enumerate(criteria)]
        gold = max(0, min(len(criteria) - 1, round(float(item.label))))
        return texts, gold

    return None


def code_token_ids(
    tokenizer: Any, codes: Iterable[str], form: str = "space"
) -> tuple[dict[str, int], list[str]]:
    """The token id of each code, and the codes that are not one token.

    ``form`` is ``"space"`` for the continuation ``" A"`` that follows
    ``Answer:`` or ``"bare"`` for ``"A"``.  A code that encodes to more than one
    id cannot be read at one position; it is returned as unscorable rather
    than substituted.
    """
    if form not in ("space", "bare"):
        raise ValueError(f"form must be 'space' or 'bare', got {form!r}")
    ids: dict[str, int] = {}
    multi: list[str] = []
    for code in codes:
        text = f" {code}" if form == "space" else code
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids[code] = encoded[0]
        else:
            multi.append(code)
    return ids, multi


def pick_code_form(tokenizers: Sequence[Any], codes: Iterable[str]) -> str:
    """The form that makes every code one token in every tokenizer.

    The space form is preferred because it is what follows ``Answer:`` in
    ordinary text.  If no form works everywhere the space form is returned and
    the per-model validity gate records the failure.
    """
    for form in ("space", "bare"):
        if all(not code_token_ids(tokenizer, codes, form)[1] for tokenizer in tokenizers):
            return form
    return "space"


def option_index_for_display(width: int, reversed_order: bool) -> list[int]:
    """The natural option index at each display position."""
    if reversed_order:
        return [width - 1 - position for position in range(width)]
    return list(range(width))


def natural_argmax(values: Sequence[float]) -> int:
    """The first maximum, with ties broken by the natural option index."""
    if not values:
        raise ValueError("natural_argmax needs at least one value")
    return max(range(len(values)), key=values.__getitem__)


def choose_option(
    code_logits: Sequence[float], display_to_option: Sequence[int]
) -> int:
    """The natural option index the readout picks.

    ``code_logits`` are in display order; they are mapped onto the natural
    option index and argmaxed there, so an exact tie resolves to the same
    option whatever the display order is. Taking the maximum in display order
    instead makes correctness depend on the order by construction, which is
    the effect the order experiments are trying to measure (#347).
    """
    if len(code_logits) != len(display_to_option):
        raise ValueError(
            f"{len(code_logits)} code logits but {len(display_to_option)} display positions"
        )
    by_option = [0.0] * len(display_to_option)
    for position, option in enumerate(display_to_option):
        by_option[option] = float(code_logits[position])
    return natural_argmax(by_option)


def has_exact_tie(values: Sequence[float]) -> bool:
    """Whether the two largest values are exactly equal.

    This is the class whose outcome the tie rule decides, so every run records
    its size. Exact equality is the test: a bf16 forward that produces 17.5
    twice is a tie, and a difference in the last place is not.
    """
    if len(values) < 2:
        return False
    ordered = sorted(values, reverse=True)
    return ordered[0] == ordered[1]


def tie_report(
    rows: Sequence[dict[str, Any]], ties_per_row: Sequence[Sequence[bool]]
) -> dict[str, Any]:
    """Tie counts and how the tie rule scored them, in one validity field.

    Args:
        rows: Prediction rows.
        ties_per_row: For each row, one flag per forward pass, true when that
            forward's two largest code logits were exactly equal.

    Returns:
        ``{"union", "per_forward", "correct"}``: the number of items tied in
        at least one forward, the same count per forward, and how many tied
        items the readout got right under the natural-index rule.
    """
    if not rows:
        return {"union": 0, "per_forward": [], "correct": 0}
    width = len(ties_per_row[0])
    per_forward = [sum(1 for flags in ties_per_row if flags[index]) for index in range(width)]
    union = sum(1 for flags in ties_per_row if any(flags))
    correct = sum(
        1 for row, flags in zip(rows, ties_per_row) if any(flags) and row["correct"]
    )
    return {"union": union, "per_forward": per_forward, "correct": correct}


def position_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Where the answers land, and whether that dominates the accuracy.

    ``rows`` carry ``pred_pos`` and ``gold_pos`` (0-based) and ``correct``.
    The bias-excluded accuracy removes items whose gold position equals the
    modal predicted position, so a model that always answers ``A`` is measured
    on the items where ``A`` is not the gold answer.  Raw and excluded accuracy
    differing by more than ten points labels the raw number position-dominated.
    """
    if not rows:
        return {
            "n": 0,
            "predicted_position_histogram": {},
            "gold_position_histogram": {},
            "accuracy_by_gold_position": {},
            "modal_predicted_position": None,
            "modal_predicted_count": 0,
            "accuracy_raw": 0.0,
            "accuracy_excluding_modal_gold": None,
            "position_dominated": False,
        }

    predicted = Counter(row["pred_pos"] for row in rows)
    gold = Counter(row["gold_pos"] for row in rows)
    by_gold: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        cell = by_gold[row["gold_pos"]]
        cell[1] += 1
        cell[0] += int(bool(row["correct"]))

    modal = max(sorted(predicted), key=lambda position: predicted[position])
    kept = [row for row in rows if row["gold_pos"] != modal]
    kept_correct = sum(int(bool(row["correct"])) for row in kept)
    raw = sum(int(bool(row["correct"])) for row in rows) / len(rows)
    excluded = kept_correct / len(kept) if kept else None
    dominated = excluded is not None and abs(raw - excluded) * 100.0 > POSITION_DOMINATED_PP

    return {
        "n": len(rows),
        "predicted_position_histogram": {
            str(position): predicted[position] for position in sorted(predicted)
        },
        "gold_position_histogram": {
            str(position): gold[position] for position in sorted(gold)
        },
        "accuracy_by_gold_position": {
            str(position): {
                "correct": cell[0],
                "total": cell[1],
                "accuracy": cell[0] / cell[1],
            }
            for position, cell in sorted(by_gold.items())
        },
        "modal_predicted_position": modal,
        "modal_predicted_count": predicted[modal],
        "accuracy_raw": raw,
        "accuracy_excluding_modal_gold": excluded,
        "position_dominated": dominated,
    }


def order_effect(
    natural: Sequence[dict[str, Any]],
    reversed_order: Sequence[dict[str, Any]],
    options_by_id: dict[str, int],
) -> dict[str, Any]:
    """Accuracy under both option orders, and how often the choice changed.

    ``pred_pos`` is the position in the order as presented, so the reversed
    prediction is mapped back to the original option index with
    ``n_options - 1 - pred_pos`` before the two orders are compared.
    """
    first = {row["id"]: row for row in natural}
    second = {row["id"]: row for row in reversed_order}
    shared = sorted(set(first) & set(second))
    if not shared:
        return {
            "n": 0,
            "accuracy_natural": 0.0,
            "accuracy_reversed": 0.0,
            "difference_pp": 0.0,
            "changed_fraction": 0.0,
        }

    changed = 0
    for item_id in shared:
        n_options = options_by_id[item_id]
        original_index = n_options - 1 - second[item_id]["pred_pos"]
        if first[item_id]["pred_pos"] != original_index:
            changed += 1

    natural_correct = sum(int(bool(first[item_id]["correct"])) for item_id in shared)
    reversed_correct = sum(int(bool(second[item_id]["correct"])) for item_id in shared)
    natural_accuracy = natural_correct / len(shared)
    reversed_accuracy = reversed_correct / len(shared)
    return {
        "n": len(shared),
        "accuracy_natural": natural_accuracy,
        "accuracy_reversed": reversed_accuracy,
        "difference_pp": (reversed_accuracy - natural_accuracy) * 100.0,
        "changed_fraction": changed / len(shared),
    }
