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

    reward_path: Optional[str]
    reward_model_saved_path: Optional[str]
    reward_model_name: str
    reward_batch_size: int
    reward_timeout: float
    local_reward_device: str
    use_http_reward: bool
    reward_service_url: Optional[str]
    reward_service_urls: Optional[List[str]]
    reward_models: Optional[List[str]]
    reward_weights: Optional[List[float]]
    component_aggregation: str
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
            reward_path=rc.reward_path,
            reward_model_saved_path=rc.reward_model_saved_path,
            reward_model_name=rc.reward_model_name,
            reward_batch_size=int(rc.reward_batch_size),
            reward_timeout=float(rc.reward_timeout),
            local_reward_device=str(rc.local_reward_device),
            use_http_reward=bool(rc.use_http_reward),
            reward_service_url=rc.reward_service_url,
            reward_service_urls=rc.reward_service_urls,
            reward_models=rc.reward_models,
            reward_weights=rc.reward_weights,
            component_aggregation=rc.component_aggregation,
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
        if self.reward_models:
            weights = self.reward_weights or []
            components = tuple(
                RewardComponentSpec(
                    model_name=str(model),
                    weight=float(weights[idx]) if idx < len(weights) else 1.0,
                )
                for idx, model in enumerate(self.reward_models)
            )
        else:
            components = (
                RewardComponentSpec(
                    model_name=str(self.reward_model_name),
                    weight=1.0,
                ),
            )
        return RewardDefinition(
            component_aggregation=str(self.component_aggregation),
            components=components,
        )

    def to_provider_config(self) -> RewardProviderConfig:
        return RewardProviderConfig(
            reward_path=self.reward_path,
            reward_model_saved_path=self.reward_model_saved_path,
            batch_size=int(self.reward_batch_size),
            timeout=float(self.reward_timeout),
        )

    def to_execution_plan(self) -> RewardExecutionPlan:
        service_urls = tuple(
            str(url)
            for url in (self.reward_service_urls or [])
            if str(url or "").strip()
        )
        service_url = (
            str(self.reward_service_url)
            if self.reward_service_url is not None and str(self.reward_service_url).strip()
            else None
        )
        if self.use_http_reward or service_urls or service_url:
            backend = "http"
        elif int(self.reward_dedicated_num_gpus) > 0 or int(self.reward_dedicated_num_nodes) > 0:
            backend = "ray_pool"
        else:
            backend = "local"
        return RewardExecutionPlan(
            location=resolve_reward_location(
                location=str(self.reward_location or "auto"),
                backend=backend,
            ),
            backend=backend,
            local_device=str(self.local_reward_device or "cpu"),
            reward_service_url=service_url,
            reward_service_urls=service_urls,
            dedicated_num_gpus=int(self.reward_dedicated_num_gpus),
            dedicated_num_nodes=int(self.reward_dedicated_num_nodes),
            dedicated_num_gpus_per_node=int(self.reward_dedicated_num_gpus_per_node),
            dedicated_gpus_per_actor=int(self.reward_dedicated_gpus_per_actor),
        )

    def component_weights(self) -> dict[str, float]:
        return self.to_definition().component_weights()


__all__ = ["RewardSchema"]
