"""Shared pytest setup.

Tests are written to degrade gracefully across the three environments they
run in (see docs in each test module):

* dev box        — no torch / no veomni / possibly no hydra: heavy tests skip
                   via ``pytest.importorskip``; pure-python tests still run.
* pod CPU venv   — torch + veomni installed, no GPU: import-level tests run.
* pod GPU        — full stack: everything runs.

The repo root is prepended to ``sys.path`` so ``unirl`` resolves even when
the package is not pip-installed (it is editable-installed on pods, where
this is a harmless no-op).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
