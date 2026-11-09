"""Attention-mass readout over option spans (issue #341).

The decision position's attention distribution is a free, permutation-
equivariant set-level signal: the mass it puts on each option's token span is a
K-vector that sums to one over the whole sequence.  This module holds the parts
that do not need torch: span alignment through the tokenizer's own offset
mapping, mass aggregation, the combination rule, and the residual diagnostics.
The script that runs the model and the tests share these definitions.

Span rule, declared before the run: an option's span is the tokens fully
contained in that option's ``{code}. {text}`` line of the prompt rendered by
:func:`model.evaluation.panel2.render_panel2`.  A token that straddles a line
boundary belongs to no span and its mass lands in the residual; the count of
such tokens is recorded in validity rather than silently attributed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from model.evaluation.panel2 import option_lines

#: The pre-registered combination: log softmax over the K code logits plus one
#: times the log span mass.  One form, fixed before the run; no sweep.
COMBINATION_LAMBDA = 1.0
MASS_FLOOR = 1e-12


def option_spans(
    tokenizer: Any,
    prompt: str,
    texts: Sequence[str],
    codes: Sequence[str],
) -> dict[str, Any]:
    """Token index spans for each option line, from the tokenizer's offsets.

    The lines are built by :func:`model.evaluation.panel2.option_lines`, the
    same function ``render_panel2`` uses, and located in the rendered prompt by
    cumulative character lengths, so no independent string matching decides
    where an option starts.  Alignment is exact: a token is in a span only when
    its character interval is fully inside that line's interval.
    """
    block = option_lines(texts, codes)
    start = prompt.find(block)
    if start < 0:
        raise ValueError("option block from option_lines is not in the prompt")
    if prompt.find(block, start + 1) != -1:
        raise ValueError("option block occurs more than once in the prompt")

    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if len(offsets) != len(encoding["input_ids"]):
        raise ValueError("offset mapping and token ids disagree in length")

    lines = block.split("\n")
    spans: list[tuple[int, int]] = []
    straddling: list[int] = []
    cursor = start
    for line in lines:
        line_start = cursor
        line_end = cursor + len(line)
        if prompt[line_start:line_end] != line:
            raise ValueError("option line does not align with the rendered prompt")
        inside = [
            index
            for index, (char_start, char_end) in enumerate(offsets)
            if char_start >= line_start and char_end <= line_end
        ]
        if not inside:
            raise ValueError(f"option line {line!r} covers no whole token")
        spans.append((inside[0], inside[-1] + 1))
        cursor = line_end + 1
    if cursor - 1 != start + len(block):
        raise ValueError("option block boundary mismatch")

    block_end = start + len(block)
    straddling = [
        index
        for index, (char_start, char_end) in enumerate(offsets)
        if char_start < block_end
        and char_end > start
        and not any(lo <= index < hi for lo, hi in spans)
    ]

    return {
        "spans": spans,
        "n_tokens": len(offsets),
        "first_span_token": spans[0][0],
        "last_span_token": spans[-1][1],
        "straddling_tokens": straddling,
        "block_char_start": start,
        "block_char_end": block_end,
    }


def span_masses(
    attention_row: Sequence[float], spans: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Mass per span, and where the rest of the distribution went.

    ``attention_row`` is the head-pooled attention from the decision position,
    one value per prompt token.  ``residual`` is mass on everything that is not
    an option span; it is split into the preamble (before the first option
    line) and the separators/other tokens inside the option block.
    """
    total = float(sum(attention_row))
    masses = [float(sum(attention_row[lo:hi])) for lo, hi in spans]
    preamble = float(sum(attention_row[: spans[0][0]]))
    span_sum = float(sum(masses))
    return {
        "masses": masses,
        "span_sum": span_sum,
        "preamble_mass": preamble,
        "residual": total - span_sum,
        "total": total,
    }


def combine_scores(
    code_logits: Sequence[float],
    masses: Sequence[float],
    lam: float = COMBINATION_LAMBDA,
    floor: float = MASS_FLOOR,
) -> list[float]:
    """The single pre-registered combination: log p_code + lambda * log mass."""
    if len(code_logits) != len(masses):
        raise ValueError("code logits and masses differ in length")
    if not code_logits:
        raise ValueError("combination needs at least one option")
    peak = max(code_logits)
    log_denominator = peak + math.log(
        sum(math.exp(value - peak) for value in code_logits)
    )
    return [
        (logit - log_denominator) + lam * math.log(max(mass, floor))
        for logit, mass in zip(code_logits, masses)
    ]


def mean_chance(options_by_id: dict[str, int]) -> float:
    """The mean uniform-guess accuracy over items with K options."""
    if not options_by_id:
        return 0.0
    return sum(1.0 / width for width in options_by_id.values()) / len(options_by_id)


def length_guard(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whether a span-mass readout is just picking the longest option.

    ``entries`` covers the variable-length items only (choice and score; a noul
    item always presents ``no``/``yes``).  Each entry carries the option texts,
    the gold natural index, and the mass readout's chosen natural index.  The
    longest option has the most characters; ties go to the lowest natural
    index.  The guard reports the longest-option baseline accuracy, the mass
    accuracy on the same items, and how often the two agree.  A span-mass
    readout that scored near this baseline would be measuring option length,
    not the decision.
    """
    longest_correct = 0
    mass_correct = 0
    agree = 0
    for entry in entries:
        texts = entry["texts"]
        if len(texts) != entry["n_options"]:
            raise ValueError("option texts and declared width disagree")
        longest = max(range(len(texts)), key=lambda index: (len(texts[index]), -index))
        longest_correct += int(longest == entry["gold"])
        mass_correct += int(entry["chosen"] == entry["gold"])
        agree += int(entry["chosen"] == longest)
    total = len(entries)
    return {
        "definition": (
            "choice and score items; longest option text by characters, "
            "ties to the lowest natural index"
        ),
        "items": total,
        "longest_option_baseline": {
            "correct": longest_correct,
            "total": total,
            "accuracy": longest_correct / total if total else 0.0,
        },
        "mass_accuracy": {
            "correct": mass_correct,
            "total": total,
            "accuracy": mass_correct / total if total else 0.0,
        },
        "agreement_with_longest": {
            "count": agree,
            "total": total,
            "fraction": agree / total if total else 0.0,
        },
    }
