"""Framework assembly config bundles built from resolved specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from diffusionrl.config.spec import (
    ModelSpec,
    PlacementSpec,
    RolloutInfo,
    SamplingSpec,
    TrainingPlan,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import RolloutActorConfig, TrainingActorConfig
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.training.types import (
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)


@dataclass(frozen=True)
class DerivedConfig:
    """Shared derived config slices reused by validation, views, and launch."""

    algorithm_dotpath: str
    model_spec: ModelSpec
    sampling_spec: SamplingSpec
    rollout_info: RolloutInfo
    train_backend_config: Any
    training_topology: TrainTopology
    train_backend_capabilities: Optional[TrainBackendCapabilities] = None
    training_plan: Optional[TrainingPlan] = None

    def require_train_backend_capabilities(self) -> TrainBackendCapabilities:
        capabilities = self.train_backend_capabilities
        if capabilities is None:
            raise ValueError(
                "DerivedConfig is missing train_backend_capabilities. "
                "Call derive_config(..., include_train_backend_capabilities=True)."
            )
        return capabilities

    def require_training_plan(self) -> TrainingPlan:
        training_plan = self.training_plan
        if training_plan is None:
            raise ValueError(
                "DerivedConfig is missing training_plan. "
                "Call attach_training_plan(...)."
            )
        return training_plan


@dataclass(frozen=True)
class RolloutLaunch:
    rollout_engine: str
    actor_gpu_requirement: int
    colocate: bool
    colocate_gpu_fraction: float
    allow_noset_multi_gpu_inference: bool
    engine_init_payload: ComponentInitPayload
    actor_init_config: RolloutActorConfig


@dataclass(frozen=True)
class TrainingLaunch:
    topology: TrainTopology
    backend_name: str
    backend_capabilities: TrainBackendCapabilities
    launch_spec: TrainBackendLaunchSpec
    colocate: bool
    colocate_gpu_fraction: float
    actor_init_config: TrainingActorConfig


@dataclass(frozen=True)
class WeightSyncSpec:
    """Typed view of all weight-sync settings needed by coordinators."""

    protocol: str
    target_modules: List[str]
    bucket_size_mb: int
    flush_cache: bool
    sync_dir: str


@dataclass(frozen=True)
class LaunchConfig:
    algorithm_init_payload: Any
    training_sampling_config: Dict[str, Any]
    rollout_info: RolloutInfo
    placement: PlacementSpec
    rollout: Optional[RolloutLaunch]
    training: TrainingLaunch
    weight_sync: WeightSyncSpec
    sampling_spec: SamplingParams


__all__ = [
    "DerivedConfig",
    "LaunchConfig",
    "PlacementSpec",
    "RolloutLaunch",
    "TrainingLaunch",
    "WeightSyncSpec",
]
