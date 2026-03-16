"""Typed reward semantics and execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RewardComponentSpec:
    """One semantic reward component."""

    model_name: str
    weight: float = 1.0


@dataclass(frozen=True)
class RewardSpec:
    """Semantic reward definition: what to compute and how to aggregate it."""

    reward_path: Optional[str]
    reward_model_saved_path: Optional[str]
    batch_size: int
    timeout: float
    component_aggregation: str
    components: Tuple[RewardComponentSpec, ...]

    @property
    def is_multi_component(self) -> bool:
        return len(self.components) > 1

    @property
    def default_model_name(self) -> str:
        if not self.components:
            return ""
        return str(self.components[0].model_name)

    @property
    def reward_models(self) -> Optional[list[str]]:
        if not self.is_multi_component:
            return None
        return [str(component.model_name) for component in self.components]

    @property
    def reward_weights(self) -> Optional[list[float]]:
        if not self.is_multi_component:
            return None
        return [float(component.weight) for component in self.components]

    def component_weights(self) -> Dict[str, float]:
        return {
            str(component.model_name): float(component.weight)
            for component in self.components
        }


@dataclass(frozen=True)
class RewardExecutionPlan:
    """Runtime deployment plan: where and with what backend rewards execute."""

    location: str
    backend: str
    local_device: str
    reward_service_url: Optional[str]
    reward_service_urls: Tuple[str, ...]
    dedicated_num_gpus: int
    dedicated_num_nodes: int
    dedicated_num_gpus_per_node: int
    dedicated_gpus_per_actor: int

    @property
    def uses_sampling_actor_execution(self) -> bool:
        return str(self.location or "manager").strip().lower() == "sampling_actor"

    @property
    def uses_http_backend(self) -> bool:
        return str(self.backend or "local").strip().lower() == "http"

    @property
    def uses_ray_backend(self) -> bool:
        return str(self.backend or "local").strip().lower() == "ray_pool"
