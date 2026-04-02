"""Typed reward config contract shared by config and runtime layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from diffusionrl.reward.spec import (
    RewardComponentSpec,
    RewardDefinition,
    RewardExecutionPlan,
    RewardProviderConfig,
    resolve_reward_location,
)


@dataclass(frozen=True)
class RewardSchema:
    """Typed view of reward-related CLI/config options."""

    reward_dotpath: Optional[str]
    reward_model_ckpt_path: Optional[str]
    reward_batch_size: int
    local_reward_device: str
    reward_backend: str
    reward_service_urls: Optional[List[str]]
    reward_components: Optional[List[str]]
    reward_weights: Optional[List[float]]
    reward_aggregation_method: str
    reward_dedicated_gpus_per_actor: int
    reward_dedicated_num_gpus: int
    reward_dedicated_num_nodes: int
    reward_dedicated_num_gpus_per_node: int
    reward_location: str

    @classmethod
    def from_args(cls, args) -> "RewardSchema":
        """Construct from TrainingArguments, delegating to the RewardConfig group."""
        rc = args.reward
        return cls(
            reward_dotpath=rc.reward_dotpath,
            reward_model_ckpt_path=rc.reward_model_ckpt_path,
            reward_batch_size=int(rc.reward_batch_size),
            local_reward_device=str(rc.local_reward_device),
            reward_backend=str(rc.reward_backend),
            reward_service_urls=rc.reward_service_urls,
            reward_components=rc.reward_components,
            reward_weights=rc.reward_weights,
            reward_aggregation_method=rc.reward_aggregation_method,
            reward_dedicated_gpus_per_actor=int(rc.reward_dedicated_gpus_per_actor),
            reward_dedicated_num_gpus=int(rc.reward_dedicated_num_gpus),
            reward_dedicated_num_nodes=int(rc.reward_dedicated_num_nodes),
            reward_dedicated_num_gpus_per_node=int(rc.reward_dedicated_num_gpus_per_node),
            reward_location=str(rc.reward_location),
        )

    @property
    def uses_sampling_actor_execution(self) -> bool:
        return self.to_execution_plan().uses_sampling_actor_execution

    @property
    def uses_driver_execution(self) -> bool:
        return self.to_execution_plan().uses_driver_execution

    def to_definition(self) -> RewardDefinition:
        raw_components = self.reward_components
        if isinstance(raw_components, str):
            component_names = [raw_components]
        elif isinstance(raw_components, list):
            component_names = list(raw_components)
        else:
            component_names = []
        weights = self.reward_weights or []
        components = tuple(
            RewardComponentSpec(
                model_name=str(component),
                weight=float(weights[idx]) if idx < len(weights) else 1.0,
            )
            for idx, component in enumerate(component_names)
            if str(component or "").strip()
        )
        return RewardDefinition(
            reward_aggregation_method=str(self.reward_aggregation_method),
            components=components,
        )

    def to_provider_config(self) -> RewardProviderConfig:
        return RewardProviderConfig(
            reward_dotpath=self.reward_dotpath,
            reward_model_ckpt_path=self.reward_model_ckpt_path,
            batch_size=int(self.reward_batch_size),
        )

    def to_execution_plan(self) -> RewardExecutionPlan:
        backend = str(self.reward_backend or "local").strip().lower()
        if backend not in {"local", "http", "ray_pool"}:
            raise ValueError(
                "reward_backend must be one of local/http/ray_pool, "
                f"got: {self.reward_backend!r}."
            )
        service_urls = tuple(
            str(url)
            for url in (self.reward_service_urls or [])
            if str(url or "").strip()
        )
        return RewardExecutionPlan(
            location=resolve_reward_location(
                location=str(self.reward_location or "auto"),
                backend=backend,
            ),
            backend=backend,
            local_device=str(self.local_reward_device or "cpu"),
            reward_service_urls=service_urls,
            dedicated_num_gpus=int(self.reward_dedicated_num_gpus),
            dedicated_num_nodes=int(self.reward_dedicated_num_nodes),
            dedicated_num_gpus_per_node=int(self.reward_dedicated_num_gpus_per_node),
            dedicated_gpus_per_actor=int(self.reward_dedicated_gpus_per_actor),
        )

    def component_weights(self) -> dict[str, float]:
        return self.to_definition().component_weights()


__all__ = ["RewardSchema"]
