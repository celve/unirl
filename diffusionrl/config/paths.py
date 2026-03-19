"""Small path helpers used by config validation/runtime checks."""

from __future__ import annotations

import os


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    # paths.py lives at diffusionrl/config/paths.py.
    # Two levels up resolves to diffusionRL repository root.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_probably_local_weight_sync_dir(path: str, *, root: str) -> bool:
    """Best-effort guard for local-only paths in multi-node checkpoint sync."""
    if not path:
        return True
    real = os.path.realpath(path)
    for prefix in ("/tmp", "/var/tmp", "/dev/shm"):
        if real == prefix or real.startswith(prefix + os.sep):
            return True
    if real == root or real.startswith(root + os.sep):
        return True
    return False


__all__ = [
    "repo_root",
    "is_probably_local_weight_sync_dir",
]
