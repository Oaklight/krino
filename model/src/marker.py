"""Marker-token selection for the option-marker readout.

The marker readout scores each option at a marker token appended to the
option's ids. Which token that is has, until now, been the tokenizer's eos,
and the marker-token ablation varies only this. The selection lives here, with
no torch import, so the override and the fallback are testable in the
pure-Python lane.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MARKER_TOKEN_KEY = "marker_token"


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


def resolve_checkpoint_marker(
    config: Mapping[str, Any], override: int | None = None
) -> int | None:
    """The marker id a staged checkpoint must be rebuilt with.

    A checkpoint trained with ``option_marker`` records the id it trained
    against.  Older checkpoints predate that field, and rebuilding one at the
    tokenizer default is silent: a head trained at 151651 would be evaluated
    at 151645 under a 151651 filename.  An absent field is therefore an error
    unless the caller passes the marker explicitly.

    Returns ``None`` when the checkpoint does not use the marker readout, and
    rejects an override in that case rather than ignoring it.
    """
    if not config.get("option_marker", False):
        if override is not None:
            raise ValueError("marker_token needs option_marker=True")
        return None
    stored = config.get(MARKER_TOKEN_KEY)
    if override is not None:
        return int(override)
    if stored is not None:
        return int(stored)
    raise ValueError(
        "residual.pt was trained with --option-marker but records no "
        "marker_token; pass the marker explicitly instead of letting the "
        "tokenizer default decide"
    )
