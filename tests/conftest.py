"""Shared pytest configuration for the unirl test suite.

Inserts the repo root on ``sys.path`` so ``import unirl`` resolves when the
package is not pip-installed into the active interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
