"""Shared engine contracts and rollout-engine type helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from diffusionrl.sde.rules import normalize_sde_type

DEDICATED_ROLLOUT_ENGINE_TYPES: Set[str] = {
    "sglang",
}


def normalize_engine_type(name: Any) -> str:
    """Normalize rollout engine type text."""

    return str(name or "").strip().lower()


def uses_dedicated_rollout_engine(name: Any) -> bool:
    """Return whether the engine runs as a dedicated rollout-side service."""

    return normalize_engine_type(name) in DEDICATED_ROLLOUT_ENGINE_TYPES


@dataclass
class EngineConfig:
    """Configuration for rollout-side inference engines."""

    model_dotpath: str = ""
    pretrained_model_ckpt_path: str = ""

    num_inference_steps: int = 50
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    guidance_scale: float = 7.5

    height: int = 256
    width: int = 256
    num_frames: int = 16

    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.sde_type = normalize_sde_type(self.sde_type)


@dataclass
class EngineCapabilities:
    """Runtime capabilities exposed by an inference engine."""

    supports_logprob: bool = True
    supports_trajectory: bool = True
    supports_prompt_embeddings: bool = True
    supports_guidance_scale: bool = True
    weight_load_mode: str = "state_dict"


__all__ = [
    "DEDICATED_ROLLOUT_ENGINE_TYPES",
    "normalize_engine_type",
    "uses_dedicated_rollout_engine",
    "EngineConfig",
    "EngineCapabilities",
]
