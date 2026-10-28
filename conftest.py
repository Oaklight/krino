"""Make the repository root importable when pytest is invoked as ``pytest tests/``.

With the default import mode pytest prepends the test file's directory
(``tests/``), not the rootdir, so ``scripts/`` and ``_vendor/`` resolve only
when something else happens to add the root. CI runs ``pytest tests/``, which
is why a test importing ``scripts.make_eval_variants`` failed there while
passing locally under ``python -m pytest`` (that form adds the cwd).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
