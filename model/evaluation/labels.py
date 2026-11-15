"""One parser for a noul label, shared by every scorer that reads one.

The label reaches the scorers in several forms: a bool from the pipeline
loaders, the integers 0 and 1, and the words ``true``/``yes``/``false``/``no``
from the public JevBench files.  Six scorers each carried their own
``str(label).lower() == "true"`` copy, which marked the string ``"yes"`` as
negative and so inverted every JevBench noul label (#299, #378).

Keep the vocabulary here rather than in any one scorer, and let the caller
decide what an unknown form means: :func:`label_polarity` returns ``None`` and
the caller either treats it as "not a polarity item" or raises.
"""

from __future__ import annotations

from typing import Any

#: The words a noul label may use for each side.
TRUE_LABELS = frozenset({"true", "yes", "1", "y"})
FALSE_LABELS = frozenset({"false", "no", "0", "n"})


def label_polarity(label: Any) -> bool | None:
    """Map a noul label to its polarity, or ``None`` when the form is unknown.

    Accepts a bool, the integers 0 and 1, and :data:`TRUE_LABELS` /
    :data:`FALSE_LABELS` in any case and with either quantity of surrounding
    space.
    """
    if isinstance(label, bool):
        return label
    if isinstance(label, int) and label in (0, 1):
        return bool(label)
    text = str(label).strip().lower()
    if text in TRUE_LABELS:
        return True
    if text in FALSE_LABELS:
        return False
    return None


def require_polarity(label: Any, item_id: Any = "?") -> bool:
    """The polarity of a noul label, raising when the form is unknown.

    A silent default to negative is how the JevBench labels were inverted, so a
    scorer that needs the answer must stop rather than guess.
    """
    value = label_polarity(label)
    if value is None:
        raise ValueError(
            f"noul label {label!r} of item {item_id!r} is neither true nor false; "
            "refusing to mark a side correct"
        )
    return value
