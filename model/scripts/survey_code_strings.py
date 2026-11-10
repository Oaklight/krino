"""Count the code space panel 2 can actually use, by code length (#350 follow-up).

Panel 2 names each option with a code that must be one token at the answer
position (``" AA"``), readable as a bare label in the option line
(``AA. text``), and not merged with the newline before it or the ``". "`` after
it. The shipped alphabet is single characters; this survey asks whether strings
of two or three letters pass those tests, which would raise the number of
options panel 2 can name without a blocking readout.

It runs no model. The result is a property of the tokenizer and the line format.

Usage::

    python model/scripts/survey_code_strings.py --snapshot <snapshot> \\
        --output results/jevbench/code_strings_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.code_space import LINE_AFTER, LINE_BEFORE, string_survey  # noqa: E402

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def config_for(args) -> dict:
    return {
        "arm": "code-space string survey",
        "alphabet": ALPHABET,
        "lengths": args.lengths,
        "line_before": LINE_BEFORE,
        "line_after": LINE_AFTER,
        "answer_form": "one leading space before the code",
        "bare_definition": "encode(code) is one token",
        "space_definition": "encode(' ' + code) is one token",
        "stable_definition": "bare and space, and encode(line_before + code) == "
        "encode(line_before) + bare, and encode(code + line_after) == bare + "
        "encode(line_after)",
        "tokenizer": args.snapshot.name,
        "add_special_tokens": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[1, 2, 3],
        help="code lengths to enumerate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/jevbench/code_strings_Qwen_Qwen3-Reranker-4B.json"),
    )
    parser.add_argument("--print-config-hash", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot))
    config = config_for(args)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    print(f"config_hash {config_hash}", flush=True)
    print(json.dumps(config, sort_keys=True, ensure_ascii=False), flush=True)
    if args.print_config_hash:
        return 0

    levels = []
    cumulative = 0
    seen_ids: dict[int, str] = {}
    collisions = []
    for length in args.lengths:
        level = string_survey(tokenizer, length, ALPHABET)
        cumulative += level["stable"]
        for code in level["stable_codes"]:
            token_id = tokenizer.encode(" " + code, add_special_tokens=False)[0]
            if token_id in seen_ids:
                collisions.append([seen_ids[token_id], code])
            seen_ids[token_id] = code
        levels.append(
            {
                "length": length,
                "candidates": level["candidates"],
                "bare": level["bare"],
                "space": level["space"],
                "both": level["both"],
                "stable": level["stable"],
                "glued": len(level["glued_both"]),
                "glued_examples": level["glued_both"][:12],
                "space_ids_distinct": level["space_ids_distinct"],
                "stable_codes": level["stable_codes"],
                "cumulative_stable": cumulative,
            }
        )
        print(
            f"length {length}: candidates {level['candidates']} bare {level['bare']} "
            f"space {level['space']} both {level['both']} stable {level['stable']} "
            f"(glued among both: {len(level['glued_both'])}) cumulative {cumulative}",
            flush=True,
        )

    result = {
        "_provenance": {"config_hash": config_hash, "config": config},
        "levels": levels,
        "effective_k": cumulative,
        "space_id_collisions": collisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"effective K from these levels: {cumulative}", flush=True)
    print(f"space-id collisions: {len(collisions)}", flush=True)
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
