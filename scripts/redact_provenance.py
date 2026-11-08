#!/usr/bin/env python3
"""Remove every absolute path from a logit run's ``_provenance`` block.

A run records where its weights and datasets lived on the machine that produced
it, and those strings are absolute.  The repository is public, so neither the
home directory nor a scratch mount may be committed.  The rule this script
enforces is **no absolute path survives** in the two fields that can carry one,
``snapshot_path`` and ``command``:

- a home root, ``/home/<user>`` or ``/Users/<user>``, becomes ``<host>``, so the
  cache tail survives:
  ``<host>/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>``;
- any other absolute root named with ``--root`` is replaced the same way;
- any remaining leading ``/`` becomes ``<host>/``, so ``/tmp/run/out.json``
  comes out as ``<host>/tmp/run/out.json``.

The tail after the root is untouched, so the cache layout, the script path and
the benchmark arguments stay readable and the revision survives.  Nothing is
special-cased by flag name: a path is a path wherever it appears.

Only those two fields are rewritten; every other field - model id, revision,
dtype, device, device_name, versions, code commit, readout, scores - must come
out byte-identical.  If a home path or an absolute path survives the rewrite in
either field, or if any other field of the payload carries a home path, the
script refuses the file rather than commit the username.

The transform is deterministic and idempotent, and it writes the same JSON shape
``eval_logit.py`` writes (no trailing newline), so an already-clean file is
byte-identical and ``--check`` is meaningful.  The write goes through a temp file
and ``os.replace``: the result file is the only copy, and a crash must not leave
it truncated.

Each file reports which of the two happened - ``rewritten`` or ``already
clean`` - and a summary gives both counts, so "nothing to do" is visibly
different from "did nothing and did not say so".

Usage:
    python scripts/redact_provenance.py model/experiments/*.json
    python scripts/redact_provenance.py --check model/experiments/*.json
    python scripts/redact_provenance.py --root /scratch/run42 out/*.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

HOST_PLACEHOLDER = "<host>"
REWRITTEN_KEYS = ("snapshot_path", "command")
HOME_ROOT = re.compile(r"(?:/home|/homes|/Users)/[A-Za-z0-9._-]+")
HOME_MARKERS = ("/home/", "/Users/")
# A ``/`` that starts a path token: not glued to a word, a dot, an existing
# placeholder, a colon (``scheme://``) or another slash (the ``//`` of a URL).
# This is the mechanical "no absolute path survives" rule; it runs after the
# roots above.  A URL is not a path and must come out byte-identical.
ABSOLUTE_LEAD = re.compile(r"(?<![\w.<>:/])/(?=[A-Za-z0-9._])")


def redact_text(text: str, extra_roots: Sequence[str] = ()) -> str:
    """Return ``text`` with every absolute root replaced by ``<host>``.

    ``extra_roots`` are absolute scratch roots found in a real file - a
    ``/scratch`` or ``/data`` mount, for instance.  They are replaced first and
    longest-first, so a nested root is consumed by the root that contains it.
    """
    for root in sorted(extra_roots, key=len, reverse=True):
        text = text.replace(root, HOST_PLACEHOLDER)
    text = HOME_ROOT.sub(HOST_PLACEHOLDER, text)
    return ABSOLUTE_LEAD.sub(f"{HOST_PLACEHOLDER}/", text)


def absolute_paths(text: str) -> list[str]:
    """Return the absolute path tokens in ``text``, for the leftover check."""
    return [
        match.group(0)
        for match in re.finditer(r"(?<![\w.<>:/])/[A-Za-z0-9._]", text)
    ]


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def absolute_home_paths(text: str) -> list[str]:
    return HOME_ROOT.findall(text)


def assert_no_home_paths(payload: Any) -> None:
    """Raise if any string anywhere in ``payload`` still carries a host root.

    The rewrite covers two fields.  This check covers the rest, so a future
    field that records a path fails loudly instead of being committed.
    """
    leftovers = sorted(
        {match for text in iter_strings(payload) for match in absolute_home_paths(text)}
    )
    if leftovers:
        raise ValueError(
            f"host roots survived redaction: {leftovers}; pass --root for a "
            "scratch root, or extend REWRITTEN_KEYS with the field that carries it"
        )


def assert_only_paths_changed(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Raise unless the only differences are the two path fields.

    The failure mode this exists for is a transform that quietly normalises
    something it should have refused - a dtype, a revision, a score.  The
    rewrite is allowed to change two keys and no others.
    """
    changed = {key for key in before if before.get(key) != after.get(key)}
    unexpected = changed - set(REWRITTEN_KEYS)
    if unexpected:
        raise AssertionError(f"rewrite touched {sorted(unexpected)}")


def sanitize_block(
    block: dict[str, Any], extra_roots: Sequence[str] = ()
) -> dict[str, Any]:
    """Return a copy of ``block`` with the two path fields redacted."""
    clean = dict(block)
    for key in REWRITTEN_KEYS:
        if not isinstance(clean.get(key), str):
            continue
        clean[key] = redact_text(clean[key], extra_roots)
        if any(marker in clean[key] for marker in HOME_MARKERS):
            raise ValueError(
                f"{key} still carries a home path after redaction: {clean[key]}"
            )
        if absolute_paths(clean[key]):
            raise ValueError(
                f"{key} still carries an absolute path after redaction: {clean[key]}"
            )
    assert_only_paths_changed(block, clean)
    return clean


def sanitize_payload(
    payload: dict[str, Any], extra_roots: Sequence[str] = ()
) -> dict[str, Any]:
    if "_provenance" not in payload:
        raise ValueError("payload has no _provenance block")
    result = dict(payload)
    result["_provenance"] = sanitize_block(payload["_provenance"], extra_roots)
    changed = {key for key in payload if payload.get(key) != result.get(key)}
    if changed - {"_provenance"}:
        raise AssertionError(f"rewrite touched {sorted(changed)}")
    assert_no_home_paths(result)
    return result


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file and ``os.replace``.

    The result file is the only copy - the GPU output is kept no backup of - and
    ``write_text`` truncates before it writes, so a crash in between would
    destroy it.
    """
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="PATH",
        help="an absolute scratch root to replace with <host>; repeatable",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any file still carries a local path",
    )
    args = parser.parse_args()
    rewritten = 0
    clean = 0
    dirty = 0
    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sanitized = sanitize_payload(payload, args.root)
        if sanitized == payload:
            clean += 1
            print(f"already clean  {path}")
            continue
        if args.check:
            dirty += 1
            print(f"LOCAL PATHS    {path}")
            continue
        write_atomically(
            path, json.dumps(sanitized, indent=2, ensure_ascii=False)
        )
        rewritten += 1
        print(f"rewritten      {path}")
    print(f"{len(args.files)} files: {rewritten} rewritten, {clean} already clean")
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
