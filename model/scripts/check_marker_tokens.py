"""Tokenization assertions before the marker-token ablation.

The marker readout scores each option at a marker token appended to the
option's ids (``model/src/listwise.py`` builds ``option_ids + [marker]``). The
ablation changes only which id is the marker, so before any run every candidate
must satisfy three properties; if any fails, the ablation stops.

1. **Independence.** A special token inserted between two option texts is one
   token of its own: tokenizing ``text + marker_string`` with
   ``add_special_tokens=False`` gives the text's ids followed by exactly the
   marker id, and the marker string before a following text gives the marker id
   followed by that text's ids. If the marker merges into a neighbour, the
   readout's position is not the option boundary.
2. **Boundary.** The marker's id is at index ``len(option_ids)``, so the last
   token of the option is immediately before it and the next option's first
   token immediately after.
3. **Length.** For K options the string form has exactly ``sum(len(ids)) + K``
   tokens, the same length as the id form the readout builds.

The six ids are the Qwen3 special tokens confirmed as exact one-token
round-trips; this script re-checks that on the tokenizer actually used.

Usage::

    python model/scripts/check_marker_tokens.py --snapshot <snapshot>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CANDIDATES = (151645, 151647, 151649, 151651, 151662, 151664)
TEXTS = [
    "card arrival",
    "top up by card charge",
    "transfer fee charged",
    "ATM support",
    "wrong exchange rate for cash withdrawal",
    "why do you need my id",
    "",
    "refund not showing up",
]


def check(tokenizer, marker_id: int) -> list[str]:
    """Return the failures for one candidate id; empty means all three hold."""
    marker = tokenizer.decode([marker_id])
    failures = []
    if not marker:
        return [f"decodes to an empty string"]
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    if marker_ids != [marker_id]:
        failures.append(f"marker string alone tokenizes to {marker_ids}, not [{marker_id}]")

    for text in TEXTS:
        base = tokenizer(text, add_special_tokens=False).input_ids
        joined = tokenizer(text + marker, add_special_tokens=False).input_ids
        if joined != base + [marker_id]:
            failures.append(f"right boundary breaks on {text!r}: {joined} vs {base + [marker_id]}")
        prefixed = tokenizer(marker + text, add_special_tokens=False).input_ids
        if prefixed != [marker_id] + base:
            failures.append(f"left boundary breaks on {text!r}: {prefixed} vs {[marker_id] + base}")

    ids = [tokenizer(text, add_special_tokens=False).input_ids for text in TEXTS]
    K = len(TEXTS)
    string_form = tokenizer(
        "".join(text + marker for text in TEXTS), add_special_tokens=False
    ).input_ids
    expected = sum(len(option) for option in ids) + K
    if len(string_form) != expected:
        failures.append(f"length {len(string_form)} for K={K}, expected {expected}")
    positions = []
    cursor = 0
    for option in ids:
        cursor += len(option)
        positions.append(cursor)
        cursor += 1
    for position in positions:
        if position >= len(string_form) or string_form[position] != marker_id:
            failures.append(f"marker not at boundary {position}")
            break
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot))
    failed = 0
    for marker_id in CANDIDATES:
        failures = check(tokenizer, marker_id)
        name = tokenizer.convert_ids_to_tokens(marker_id)
        status = "PASS" if not failures else "FAIL"
        print(f"{marker_id} {name!r}: {status}", flush=True)
        for failure in failures:
            print(f"    {failure}", flush=True)
        failed += bool(failures)
    if failed:
        print(f"{failed} candidate(s) failed; stop before any run", flush=True)
        return 1
    print(f"all {len(CANDIDATES)} candidates pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
