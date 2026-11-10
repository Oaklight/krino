"""Marker-token selection for the option-marker readout.

The marker readout scores each option at a marker token appended to the
option's ids. Which token that is has, until now, been the tokenizer's eos,
and the marker-token ablation varies only this. The selection lives here, with
no torch import, so the override and the fallback are testable in the
pure-Python lane.
"""

from __future__ import annotations

from typing import Any


def marker_token_id_for(tokenizer: Any, override: int | None = None) -> int:
    """The option-marker token id: the override, else eos, sep or pad.

    With no override the shipped behaviour is unchanged. The fallback order is
    eos, sep, pad; a tokenizer with none of them cannot host the readout.
    """
    if override is not None:
        return int(override)
    marker = (
        getattr(tokenizer, "eos_token_id", None)
        or getattr(tokenizer, "sep_token_id", None)
        or getattr(tokenizer, "pad_token_id", None)
    )
    if marker is None:
        raise ValueError(
            "option_marker needs a tokenizer with an eos, sep or pad token "
            "to use as the option marker"
        )
    return int(marker)
