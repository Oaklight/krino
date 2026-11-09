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

#: Declared prompt orderings for #336.  ``shipped`` is ``PANEL2_TEMPLATE``.
#: ``options_first`` moves the question block after the options block and
#: changes nothing else; ``sandwich`` repeats the question block after the
#: options, so it differs in content and token count as well as position and
#: is a repetition treatment, not a third point on an order axis.
ORDER_TEMPLATES = {
    "shipped": PANEL2_TEMPLATE,
    "options_first": (
        "State: {state}\n"
        "Options:\n"
        "{options}\n"
        "Question: {instructions}\n"
        "Answer:"
    ),
    "sandwich": (
        "State: {state}\n"
        "Question: {instructions}\n"
        "Options:\n"
        "{options}\n"
        "Question: {instructions}\n"
        "Answer:"
    ),
}

POSITION_DOMINATED_PP = 10.0


def read_specs(n_layers: int) -> dict[str, tuple[str, int] | tuple[str]]:
    """The five declared read configurations for #337.

    ``n_layers`` is the backbone's block count.  The three middle reads are
    one third, two thirds, and last-but-four *by integer division of the block
    count*, so the declaration derives from the architecture rather than from a
    hand-picked layer.  ``last`` is the shipped read and returns the model's
    own next-token logits; ``mean_last_n=4`` averages the last four raw block
    outputs the hidden-state tuple exposes, excluding the final normed entry.
    """
    if n_layers < 8:
        raise ValueError(f"n_layers must be at least 8, got {n_layers}")
    return {
        "last": ("last",),
        f"layer={n_layers // 3}": ("layer", n_layers // 3),
        f"layer={2 * n_layers // 3}": ("layer", 2 * n_layers // 3),
        f"layer={n_layers - 4}": ("layer", n_layers - 4),
        "mean_last_n=4": ("mean_last_n", 4),
    }


def parse_read_spec(text: str, n_layers: int) -> tuple[str, int] | tuple[str]:
    """One declared read configuration by name."""
    specs = read_specs(n_layers)
    if text not in specs:
        raise ValueError(
            f"unknown read config {text!r}; expected one of {sorted(specs)}"
        )
    return specs[text]


def check_read_layout(model: Any, output: Any) -> None:
    """Fail loudly if the hidden-state layout is not the one the reads assume.

    Verified on transformers 5.21: the tuple has ``n_layers + 1`` entries,
    index 0 is the embedding output, index k is the output of block k for
    ``1 <= k <= n_layers - 1``, and the last entry is the final normed state
    rather than block ``n_layers``'s raw output.  The last entry must be the
    tensor the LM head reads; if a library change breaks that, a middle-layer
    read would silently mean a different layer.
    """
    import torch

    hidden_states = output.hidden_states
    n_layers = int(model.config.num_hidden_layers)
    if len(hidden_states) != n_layers + 1:
        raise RuntimeError(
            f"hidden-state tuple has {len(hidden_states)} entries for "
            f"{n_layers} blocks; the read layout no longer matches read_specs"
        )
    if not torch.equal(model.lm_head(hidden_states[-1]), output.logits):
        raise RuntimeError(
            "the last hidden state is not the tensor the LM head reads; "
            "hidden-state semantics changed"
        )


def read_logits(
    model: Any,
    hidden_states: Sequence[Any],
    spec: tuple[str, int] | tuple[str],
    last_logits: Any,
) -> Any:
    """Logits from one declared read of the final position's hidden states.

    ``last`` returns the model's own next-token logits unchanged.  A
    middle-layer read applies the final norm and the LM head to a
    representation that norm was never trained for: the logit-lens caveat.
    ``mean_last_n`` averages raw block outputs and excludes the final normed
    entry, so the average is over the last n blocks the tuple exposes.
    """
    import torch

    kind = spec[0]
    if kind == "last":
        return last_logits
    if kind == "layer":
        hidden = hidden_states[spec[1]][0, -1]
    elif kind == "mean_last_n":
        hidden = torch.stack(
            [state[0, -1] for state in hidden_states[-(spec[1] + 1) : -1]]
        ).mean(0)
    else:
        raise ValueError(f"unknown read spec {spec!r}")
    return model.lm_head(model.model.norm(hidden)).float()


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
    state: str,
    instructions: str,
    texts: Sequence[str],
    codes: Sequence[str],
    order: str = "shipped",
) -> str:
    """The prompt form for one declared ordering (#336).

    ``order`` must be a key of :data:`ORDER_TEMPLATES`; the default is the
    shipped order every panel-2 cell before #336 used.
    """
    if order not in ORDER_TEMPLATES:
        raise ValueError(
            f"unknown order {order!r}; expected one of {sorted(ORDER_TEMPLATES)}"
        )
    return ORDER_TEMPLATES[order].format(
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
