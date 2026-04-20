"""Typed reward config contract: spec types and schema shared by config and runtime layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Spec types (reward semantics, provider config, execution plan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardComponentSpec:
    """One semantic reward component."""

    model_name: str
    weight: float = 1.0


@dataclass(frozen=True)
class RewardDefinition:
    """Semantic reward definition: what to compute and how to aggregate it."""

    reward_aggregation_method: str
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
    def component_names(self) -> Optional[list[str]]:
        if not self.is_multi_component:
            return None
        return [str(component.model_name) for component in self.components]

    @property
    def component_weights_list(self) -> Optional[list[float]]:
        if not self.is_multi_component:
            return None
        return [float(component.weight) for component in self.components]

    @property
    def reward_models(self) -> Optional[list[str]]:
        """Backward-compatible alias for ``component_names``."""
        return self.component_names

    @property
    def reward_weights(self) -> Optional[list[float]]:
        """Backward-compatible alias for ``component_weights_list``."""
        return self.component_weights_list

    def component_weights(self) -> Dict[str, float]:
        return {str(component.model_name): float(component.weight) for component in self.components}


@dataclass(frozen=True)
class RewardProviderConfig:
    """Provider/scorer configuration: which scorer to load and with what limits."""

    reward_dotpath: Optional[str]
    reward_model_ckpt_path: Optional[str]
    batch_size: int
    timeout: float = 60.0


@dataclass(frozen=True)
class RewardExecutionPlan:
    """Runtime deployment plan: where and with what backend rewards execute."""

    backend: str
    local_device: str
    reward_service_urls: Tuple[str, ...]

    @property
    def uses_http_backend(self) -> bool:
        return str(self.backend or "local").strip().lower() == "http"


# ---------------------------------------------------------------------------
# RewardSpec (config → typed view)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardSpec:
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

    @classmethod
    def from_args(cls, args) -> "RewardSpec":
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
        )

    def to_definition(self) -> RewardDefinition:
        raw_components = self.reward_components
        if isinstance(raw_components, list):
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
        if backend not in {"local", "http"}:
            raise ValueError(f"reward_backend must be one of local/http, got: {self.reward_backend!r}.")
        service_urls = tuple(str(url) for url in (self.reward_service_urls or []) if str(url or "").strip())
        return RewardExecutionPlan(
            backend=backend,
            local_device=str(self.local_reward_device or "cpu"),
            reward_service_urls=service_urls,
        )

    def component_weights(self) -> dict[str, float]:
        return self.to_definition().component_weights()


__all__ = [
    "RewardComponentSpec",
    "RewardDefinition",
    "RewardExecutionPlan",
    "RewardProviderConfig",
    "RewardSpec",
]
