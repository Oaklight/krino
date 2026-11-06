"""Environment kill switches for mutation-checked features.

A kill switch turns a guarded feature off at runtime, so a test can be re-run
with the feature disabled.  ``tests/conftest.py`` uses these switches: a test
that still passes with its feature off asserts nothing about that feature.

The switches exist for tests, not for production.  An unset, empty, or
false-like value leaves the feature on.
"""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})


def feature_disabled(name: str) -> bool:
    """Return whether the environment kill switch ``name`` is on.

    Args:
        name: Environment variable name, e.g. ``KRINO_DISABLE_ANSWERABLE_MASK``.

    Returns:
        True when the variable holds ``1``, ``true``, ``yes`` or ``on``
        (case-insensitive); False otherwise.
    """
    return os.environ.get(name, "").strip().lower() in _TRUE
