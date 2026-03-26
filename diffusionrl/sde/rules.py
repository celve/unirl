"""Canonical transition-rule helpers for shared SDE math."""

from __future__ import annotations

from typing import Optional, Tuple

CANONICAL_SDE_TYPES: Tuple[str, ...] = ("flow", "cps", "dance", "dpm2")
SUPPORTED_USER_SDE_TYPES: Tuple[str, ...] = CANONICAL_SDE_TYPES


def normalize_sde_type(sde_type: Optional[str], *, default: str = "flow") -> str:
    """Normalize transition rule names to stable canonical lowercase values."""

    raw = str(sde_type or "").strip().lower()
    fallback = str(default or "flow").strip().lower() or "flow"
    if not raw:
        raw = fallback
    return raw


def is_deterministic_sde_type(
    sde_type: Optional[str],
    eta: Optional[float] = None,
    *,
    default: str = "flow",
) -> bool:
    """Whether the transition is deterministic at runtime."""

    normalized = normalize_sde_type(sde_type, default=default)
    if normalized == "dpm2":
        return True
    if eta is None:
        return False
    return float(eta) == 0.0


def supported_sde_type_text(values: Tuple[str, ...] = SUPPORTED_USER_SDE_TYPES) -> str:
    """Render a stable list of accepted transition rule names."""

    return ", ".join(values)


__all__ = [
    "CANONICAL_SDE_TYPES",
    "SUPPORTED_USER_SDE_TYPES",
    "normalize_sde_type",
    "is_deterministic_sde_type",
    "supported_sde_type_text",
]
