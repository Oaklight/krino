"""Resolve the provenance block a benchmark run embeds in its output JSON.

The September Phase A files carry no model revision, dtype, device, code commit
or command, so their numbers cannot be audited.  Every new run records those
fields at the top level under ``_provenance``.

Nothing here imports torch, so the block can be built and tested without the
model stack installed.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # a broken or partially built package must not fail the run
        return False
    return True


def environment_info() -> dict[str, str]:
    """Record the interpreter and the libraries the numbers depend on.

    ``device`` is ``cuda:0`` on every machine and every card, so it cannot say
    which environment produced a file.  Versions can, and torch is imported
    here rather than at module scope so this module stays importable - and
    testable - without the model stack installed.

    ``kernel_path`` records whether the GDN kernels ran fused or by reference.
    Qwen3.5's linear-attention layers use ``flash-linear-attention`` and
    ``causal_conv1d`` when both are importable and fall back to PyTorch
    otherwise; the two paths compute the same quantity, but a bf16 reduction
    can differ in the last bits, so a row has to say which one produced it.
    """
    info: dict[str, str] = {"python_version": platform.python_version()}
    try:
        import torch
    except (ImportError, OSError):
        info["torch_version"] = "unavailable"
        info["device_name"] = "unavailable"
    else:
        info["torch_version"] = str(torch.__version__)
        try:
            available = torch.cuda.is_available()
        except Exception:  # a broken driver must not fail the run's record
            available = False
        if available:
            try:
                info["device_name"] = torch.cuda.get_device_name()
            except Exception:
                info["device_name"] = "cuda (name unavailable)"
        else:
            info["device_name"] = "cpu"
    try:
        import transformers
    except (ImportError, OSError):
        info["transformers_version"] = "unavailable"
    else:
        info["transformers_version"] = str(transformers.__version__)
    fused_kernels = {
        "fla": _importable("fla"),
        "causal_conv1d": _importable("causal_conv1d"),
    }
    for name, present in fused_kernels.items():
        info[name] = "available" if present else "absent"
    info["kernel_path"] = "fused" if all(fused_kernels.values()) else "reference"
    return info


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


def readout_label(strategy: str, norm: str) -> str:
    """Name the readout in the words the JevBench and logit tables use.

    ``description`` scores the full option text as a continuation, ``label``
    scores the single letter or digit that names the option.  The two differ by
    up to five points at high option counts, so a row must say which it is.
    """
    if strategy == "description":
        return f"option-text continuation, {norm}-normalised log-prob"
    if strategy == "label":
        return f"one-token label code, {norm}-normalised log-prob"
    return f"strategy={strategy}, {norm}-normalised log-prob"


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
        "readout": readout_label(strategy, norm),
        "strategy": strategy,
        "norm": norm,
        "command": " ".join(command if command is not None else sys.argv),
        **environment_info(),
    }
    if variant:
        block["variant"] = variant
    if extra:
        block.update(extra)
    return block
