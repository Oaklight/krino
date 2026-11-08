"""Resolve the provenance block a benchmark run embeds in its output JSON.

The September Phase A files carry no model revision, dtype, device, code commit
or command, so their numbers cannot be audited.  Every new run records those
fields at the top level under ``_provenance``.

Nothing here imports torch, so the block can be built and tested without the
model stack installed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def resolve_code_commit(root: str | Path, explicit: str | None = None) -> str:
    """Return the commit the run executes from.

    ``explicit`` wins when given.  Otherwise the tree is asked; a tree that is
    not a git checkout returns ``"unresolved"`` rather than raising, so an
    exported copy still produces a file that says what it could not record.
    """
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unresolved"
    return result.stdout.strip() or "unresolved"


def resolve_snapshot_revision(snapshot_path: str | Path) -> str:
    """Return the commit sha a Hugging Face cache snapshot path encodes.

    ``snapshot_download`` returns ``.../snapshots/<sha>``; the last path
    component is the resolved revision.
    """
    return Path(snapshot_path).name


def build_run_provenance(
    *,
    model_id: str,
    snapshot_path: str | Path,
    dtype: str,
    device: str,
    strategy: str,
    norm: str,
    root: str | Path,
    variant: str | None = None,
    code_commit: str | None = None,
    command: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``_provenance`` block for one run.

    ``model_id`` is the repository id as written on Hugging Face, never a tag
    alias; ``revision`` is the resolved commit taken from the snapshot path, so
    the two together name the weights exactly.  ``variant`` records whether the
    checkpoint is an instruct/SFT build or a base one.
    """
    block: dict[str, Any] = {
        "model_id": model_id,
        "revision": resolve_snapshot_revision(snapshot_path),
        "snapshot_path": str(snapshot_path),
        "dtype": dtype,
        "device": device,
        "code_commit": resolve_code_commit(root, code_commit),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy": strategy,
        "norm": norm,
        "command": " ".join(command if command is not None else sys.argv),
    }
    if variant:
        block["variant"] = variant
    if extra:
        block.update(extra)
    return block
