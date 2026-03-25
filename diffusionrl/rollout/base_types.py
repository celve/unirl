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


@dataclass(frozen=True)
class RewardHookResult:
    """Reward hook output consumed by the default rollout pipeline."""

    rewards: torch.Tensor
    reward_components: Dict[str, List[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutFunctionResult:
    """Normalized rollout-stage output before driver-side advantage/assembly."""

    request: RolloutRequest
    sampler_outputs: List[RolloutSamples]
    rewards: torch.Tensor
    reward_components: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
