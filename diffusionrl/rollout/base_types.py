"""Typed contracts for rollout extension hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

from diffusionrl.types.sampling import RolloutRequest, RolloutSamples


@dataclass(frozen=True)
class RolloutContext:
    """Execution context passed into one rollout function invocation."""

    rollout_id: int
    sde_indices: Optional[Set[int]] = None
    collect_media_preview: bool = False
    media_max_items: int = 8
    debug_trace: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, init=False)
class RewardHookResult:
    """Reward hook output consumed by the default rollout pipeline."""

    rewards: torch.Tensor
    component_rewards: Dict[str, List[float]] = field(default_factory=dict)

    def __init__(
        self,
        *,
        rewards: torch.Tensor,
        component_rewards: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(
            self,
            "component_rewards",
            {
                str(name): list(values or [])
                for name, values in dict(component_rewards or {}).items()
            },
        )


@dataclass(frozen=True, init=False)
class RolloutFunctionResult:
    """Normalized rollout-stage output before driver-side advantage/assembly."""

    request: RolloutRequest
    sampler_outputs: List[RolloutSamples]
    rewards: torch.Tensor
    component_rewards: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        request: RolloutRequest,
        sampler_outputs: List[RolloutSamples],
        rewards: torch.Tensor,
        component_rewards: Optional[Dict[str, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "sampler_outputs", list(sampler_outputs or []))
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(
            self,
            "component_rewards",
            {
                str(name): list(values or [])
                for name, values in dict(component_rewards or {}).items()
            },
        )
        object.__setattr__(self, "metadata", dict(metadata or {}))
