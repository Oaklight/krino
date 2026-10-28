"""Catch references to helpers that no longer exist in eval_calibration.py.

The script imports torch at module scope, so CI never imports it and a
NameError in a rarely taken branch survives review. That is how
`--predictions` shipped calling `_argmax`, a helper the tie-stable change had
removed: the flag was exercised for the first time on a GPU, forty minutes into
a run.

This is a static scan, not an import: it resolves nothing, and it is
deliberately lenient (every bound name anywhere counts as defined), so it
reports only names that appear nowhere in the file.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "model" / "scripts" / "eval_calibration.py"


def _undefined_names(source: str) -> list[str]:
    tree = ast.parse(source)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs:
                defined.add(arg.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in defined
            and not hasattr(builtins, node.id)
        }
    )


def test_no_reference_to_a_removed_helper():
    undefined = _undefined_names(SCRIPT.read_text(encoding="utf-8"))
    assert undefined == ["__file__"] or undefined == [], undefined


def test_the_scan_catches_a_removed_helper():
    """Guard: the checker must fail on the shape it was written for."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_argmax(" in source or "tie_stable_argmax(" in source
    broken = source.replace(
        'prediction = record["labels"][\n'
        '                        tie_stable_argmax(probabilities, record["label_order"])\n'
        '                    ]',
        'prediction = record["labels"][_argmax(probabilities)]',
    )
    if broken == source:  # the formatting moved; scan a synthetic case instead
        broken = source + "\n\ndef _probe():\n    return _argmax([0.1])\n"
    assert "_argmax" in _undefined_names(broken)
