"""Marker-token selection for the option-marker readout.

The marker readout scores each option at a marker token appended to the
option's ids. Which token that is has, until now, been the tokenizer's eos,
and the marker-token ablation varies only this. The selection lives here, with
no torch import, so the override and the fallback are testable in the
pure-Python lane.
"""

from __future__ import annotations

import sys
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
        if stored is not None and int(stored) != int(override):
            # The swap experiment legitimately overrides a recorded id, so
            # this is a warning, not an error; the mismatch must be visible.
            print(
                "  warning: checkpoint records marker_token %s but "
                "marker_token %s was passed; using %s"
                % (stored, override, override),
                file=sys.stderr,
                flush=True,
            )
        return int(override)
    if stored is not None:
        return int(stored)
    if MARKER_TOKEN_KEY in config:
        raise ValueError(
            "residual.pt records marker_token: null, which means the save did "
            "not resolve the marker id; pass the marker explicitly"
        )
    raise ValueError(
        "residual.pt was trained with --option-marker but records no "
        "marker_token; pass the marker explicitly instead of letting the "
        "tokenizer default decide"
    )


def resolve_training_marker(args: Any, tokenizer: Any) -> int | None:
    """The marker id the checkpoint will record, resolved at save time.

    ``--marker-token`` is optional; omitting it means the shipped default, not
    "unknown".  The checkpoint must therefore store the id that was actually
    used (151645 for the shipped tokenizer), not the raw flag, or a default
    arm would be indistinguishable from a legacy file that predates the field.
    """
    if not args.option_marker:
        return None
    return marker_token_id_for(tokenizer, args.marker_token)


def checkpoint_config(
    args: Any, answerable_pos_weight: float, marker_token: int | None
) -> dict:
    """The config stored beside the heads in ``residual.pt``.

    This is the checkpoint's own record, not the recipe hash.  ``marker_token``
    is the resolved id, so a consumer never falls back to the tokenizer
    default, which is silent and would evaluate a non-default arm at eos.  The
    key is deliberately excluded from ``train_residual.recipe_config``, which
    keeps the raw flag, so the six arm hashes are unchanged.

    It lives here, beside :func:`resolve_checkpoint_marker`, so the producer
    and the consumer of the same schema share one torch-free module.
    """
    return {
        "backbone": args.model,
        "rank": args.rank,
        "max_length": args.max_length,
        "kl_weight": args.kl_weight,
        "ce_weight": args.ce_weight,
        "seed": args.seed,
        "option_marker": args.option_marker,
        "noul_marker": args.noul_marker,
        "marker_token": marker_token,
        "option_chunk_size": args.option_chunk_size if args.option_marker else None,
        "option_max_tokens": args.option_max_tokens if args.option_marker else None,
        "option_batch": args.option_batch if args.option_marker else None,
        "train_safe_buckets": args.train_safe_buckets,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank if args.lora_layers is not None else None,
        "lora_alpha": args.lora_alpha if args.lora_layers is not None else None,
        "answerable_head": args.answerable_head,
        "answerable_pos_weight": answerable_pos_weight,
    }
