"""Loader for the development set (#298).

The development set is a committed Jsonl of ``TypedQuestion`` rows, selected by
``model/experiments/dev_set/build_dev_set.py``. It exists because JevBench
public's 231 items resolve about 8 pp and the effects the next stage targets are
2-3 pp. It is a development instrument only: JevBench public stays the
acceptance gate, and no JevBench number is ever inferred from this set.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterator

from data.format import TypedQuestion

_FIELDS = {field.name for field in dataclasses.fields(TypedQuestion)}


def load_dev_set(path: Path | str) -> Iterator[TypedQuestion]:
    """Yield the committed development items in file order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"development set not found at {path}; build it with "
            "model/experiments/dev_set/build_dev_set.py (#298)"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            unknown = set(payload) - _FIELDS
            if unknown:
                raise ValueError(f"unknown fields {sorted(unknown)} on item {payload.get('id')!r}")
            yield TypedQuestion(**payload)
