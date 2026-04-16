"""Canonical transition-rule helpers for shared SDE math."""

from __future__ import annotations

from typing import Optional, Tuple

CANONICAL_SDE_TYPES: Tuple[str, ...] = ("flow", "cps", "dance", "dpm2")
SUPPORTED_USER_SDE_TYPES: Tuple[str, ...] = CANONICAL_SDE_TYPES


def is_deterministic_sde_type(
    sde_type: str,
    eta: Optional[float] = None,
) -> bool:
    """Whether the transition is deterministic at runtime.

    Expects *sde_type* to already be canonical lowercase (validated by
    ``_validate_metadata_choices`` in config validation).
    """
    if sde_type == "dpm2":
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
    "is_deterministic_sde_type",
    "supported_sde_type_text",
]
