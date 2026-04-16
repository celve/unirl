"""Shared engine declarations and rollout-engine type helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Set

ROLLOUT_ENGINE_TYPES: Set[str] = {
    "sglang",
}


def normalize_engine_type(name: Any) -> str:
    """Normalize rollout engine type text."""

    return str(name or "").strip().lower()


def uses_dedicated_rollout_engine(name: Any) -> bool:
    """Return whether a normalized engine type uses a dedicated rollout service."""

    return isinstance(name, str) and name in ROLLOUT_ENGINE_TYPES


@dataclass
class EngineCapabilities:
    """Runtime capabilities exposed by an inference engine."""

    supports_logprob: bool = True
    supports_trajectory: bool = True
    supports_prompt_embeddings: bool = True
    supports_guidance_scale: bool = True
    weight_load_mode: str = "state_dict"


__all__ = [
    "ROLLOUT_ENGINE_TYPES",
    "normalize_engine_type",
    "uses_dedicated_rollout_engine",
    "EngineCapabilities",
]
