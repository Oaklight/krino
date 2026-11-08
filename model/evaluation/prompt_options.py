"""Option text for a noul question, shared by every caller.

``LogitScorer._eval_noul`` builds " yes" / " no" from the question's own
criteria when it has them, because JevBench's 74 noul items each phrase the two
answers differently.  The JevBench screen must use the same wording or its noul
column is measured under a prompt the table does not claim.  The construction
lives here, torch-free, so both callers use one definition and a test can check
it without the model stack.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TRUE = "yes"
DEFAULT_FALSE = "no"


def noul_option_texts(criteria: Any) -> list[str]:
    """Return the scored option texts for a noul question, true first.

    ``criteria`` is the question's ``{"true": ..., "false": ...}`` mapping when
    it carries one.  A missing or partial mapping falls back to yes/no per
    side, which is what the scorer has always done.
    """
    true_desc = DEFAULT_TRUE
    false_desc = DEFAULT_FALSE
    if isinstance(criteria, dict):
        true_desc = criteria.get("true") or DEFAULT_TRUE
        false_desc = criteria.get("false") or DEFAULT_FALSE
    return [f" {true_desc}", f" {false_desc}"]
