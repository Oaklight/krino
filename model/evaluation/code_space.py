"""The single-token code space, and the tiers a code alphabet can be drawn from.

Panel 2 names each option with a one-token code. The shipped alphabet is 84
characters; the tokenizer has thousands more single-character tokens. This
module answers two questions without knowing anything about the model, so the
answer is a property of the tokenizer and can be checked:

* which characters are one token **bare** (the option line ``{code}. {text}``)
  and one token with a **leading space** (the answer position ``" A"``), and
  which of those also survive their neighbours without merging;
* which tier a character belongs to, so an experiment can extend the alphabet
  one tier at a time instead of in one unreadable jump.

Everything takes a tokenizer-like object with ``encode``/``decode``/``get_vocab``
and does not import torch.
"""

from __future__ import annotations

import random
import unicodedata
from typing import Any

# The prompt context the code sits in, and the separator that follows it.
LINE_BEFORE = "\nOptions:\n"
LINE_AFTER = ". "

TIER_ORDER = (
    "ascii_alnum",
    "ascii_symbol",
    "other_letter",
    "hangul",
    "cjk_ideograph",
    "cjk_punct_fullwidth",
    "kana",
    "other_symbol",
    "other_punct",
    "other_number",
    "combining_mark",
    "space_control",
)

# Tiers a code alphabet should not draw from: whitespace, control characters,
# marks that attach to the previous character and cannot stand alone.
EXCLUDED_TIERS = frozenset({"space_control", "combining_mark"})


def tier_of(ch: str) -> str:
    """The tier a single character belongs to."""
    code = ord(ch)
    if ch.isascii() and (ch.isalpha() or ch.isdigit()):
        return "ascii_alnum"
    if ch.isascii() and 0x21 <= code <= 0x7E:
        return "ascii_symbol"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
        return "cjk_ideograph"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "kana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
        return "cjk_punct_fullwidth"
    category = unicodedata.category(ch)
    if category in ("Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"):
        return "space_control"
    if category.startswith("L"):
        return "other_letter"
    if category.startswith("N"):
        return "other_number"
    if category.startswith("S"):
        return "other_symbol"
    if category.startswith("P"):
        return "other_punct"
    if category.startswith("M"):
        return "combining_mark"
    return "other"


def code_forms(tokenizer: Any, ch: str) -> dict[str, Any]:
    """Every token-level fact about one candidate code character."""
    bare = tokenizer.encode(ch, add_special_tokens=False)
    spaced = tokenizer.encode(" " + ch, add_special_tokens=False)
    before = tokenizer.encode(LINE_BEFORE, add_special_tokens=False)
    after = tokenizer.encode(LINE_AFTER, add_special_tokens=False)
    left = tokenizer.encode(LINE_BEFORE + ch, add_special_tokens=False)
    right = tokenizer.encode(ch + LINE_AFTER, add_special_tokens=False)
    bare_one = len(bare) == 1
    space_one = len(spaced) == 1
    return {
        "char": ch,
        "tier": tier_of(ch),
        "bare": bare_one,
        "space": space_one,
        "left_ok": bare_one and left == before + bare,
        "right_ok": bare_one and right == bare + after,
        "bare_id": bare[0] if bare_one else None,
        "space_id": spaced[0] if space_one else None,
    }


def single_char_tokens(tokenizer: Any) -> list[str]:
    """Every character that has a one-character token, bare or space-prefixed."""
    found: set[str] = set()
    added = set(tokenizer.get_added_vocab())
    special = set(tokenizer.all_special_ids)
    for token, index in tokenizer.get_vocab().items():
        if token in added or index in special:
            continue
        text = tokenizer.decode([index], skip_special_tokens=False)
        if len(text) == 1:
            found.add(text)
        elif len(text) == 2 and text[0] == " ":
            found.add(text[1])
    return sorted(found)


def survey(tokenizer: Any, chars: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """``code_forms`` for every candidate character, keyed by character."""
    characters = chars if chars is not None else single_char_tokens(tokenizer)
    return {ch: code_forms(tokenizer, ch) for ch in characters}


def usable_chars(
    survey_rows: dict[str, dict[str, Any]],
    *,
    form: str = "space",
    context_stable: bool = True,
    exclude: frozenset[str] = EXCLUDED_TIERS,
) -> dict[str, list[str]]:
    """Usable characters per tier under one code form.

    ``form="space"`` requires the answer-position form to be one token;
    ``form="bare"`` requires only the bare form. ``context_stable`` additionally
    requires that the newline before and the ``". "`` after do not merge with
    the code.
    """
    if form not in ("space", "bare"):
        raise ValueError(f"form must be 'space' or 'bare', got {form!r}")
    by_tier: dict[str, list[str]] = {}
    for ch, row in survey_rows.items():
        if row["tier"] in exclude:
            continue
        if not row["bare"]:
            continue
        if form == "space" and not row["space"]:
            continue
        if context_stable and not (row["left_ok"] and row["right_ok"]):
            continue
        by_tier.setdefault(row["tier"], []).append(ch)
    return {tier: sorted(chars) for tier, chars in sorted(by_tier.items())}


def sample_tier(
    survey_rows: dict[str, dict[str, Any]],
    tier: str,
    count: int,
    seed: int,
    *,
    form: str = "space",
    context_stable: bool = True,
    name_prefix: str | None = None,
) -> list[str]:
    """``count`` characters from one tier, deterministically, sorted by codepoint.

    The sample is drawn from the sorted list with a seeded generator and then
    sorted again, so the result depends only on (tier, count, seed, tokenizer)
    and can be written into a pre-registration before any run. ``name_prefix``
    narrows the pool by Unicode character name, so a script-specific tier can be
    drawn from the generic ``other_letter`` category ("LATIN ..." for accented
    Latin, excluding Greek, Hebrew and the rest).
    """
    pool = usable_chars(survey_rows, form=form, context_stable=context_stable).get(
        tier, []
    )
    if name_prefix is not None:
        pool = [
            ch
            for ch in pool
            if unicodedata.name(ch, "").startswith(name_prefix)
        ]
    if len(pool) < count:
        raise ValueError(f"tier {tier!r} has {len(pool)} usable chars, need {count}")
    chosen = random.Random(seed).sample(pool, count)
    return sorted(chosen)
