"""Reward-domain schema shared by reward service and validation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RewardSchema:
    """Typed view of reward-related CLI/config options."""

    reward_path: str
    reward_model_saved_path: Optional[str]
    reward_model_name: str
    reward_batch_size: int
    reward_timeout: float
    use_http_reward: bool
    reward_service_url: Optional[str]
    reward_service_urls: Optional[List[str]]
    reward_models: Optional[List[str]]
    reward_weights: Optional[List[float]]
    reward_aggregation: str
    reward_mix_mode: str
    reward_dedicated_gpus_per_actor: int
    reward_dedicated_num_gpus: int
    reward_dedicated_num_nodes: int
    reward_dedicated_num_gpus_per_node: int
    reward_placement_strategy: str

    @classmethod
    def from_args(cls, args) -> "RewardSchema":
        return cls(
            reward_path=getattr(
                args,
                "reward_path",
                "diffusionrl.reward.local.LocalRewardWorker",
            ),
            reward_model_saved_path=getattr(args, "reward_model_saved_path", None),
            reward_model_name=getattr(args, "reward_model_name", "hpsv2"),
            reward_batch_size=int(getattr(args, "reward_batch_size", 8)),
            reward_timeout=float(getattr(args, "reward_timeout", 60.0)),
            use_http_reward=bool(getattr(args, "use_http_reward", False)),
            reward_service_url=getattr(args, "reward_service_url", None),
            reward_service_urls=getattr(args, "reward_service_urls", None),
            reward_models=getattr(args, "reward_models", None),
            reward_weights=getattr(args, "reward_weights", None),
            reward_aggregation=getattr(args, "reward_aggregation", "weighted_sum"),
            reward_mix_mode=getattr(args, "reward_mix_mode", "reward_aggr"),
            reward_dedicated_gpus_per_actor=int(
                getattr(args, "reward_dedicated_gpus_per_actor", 1)
            ),
            reward_dedicated_num_gpus=int(getattr(args, "reward_dedicated_num_gpus", 0)),
            reward_dedicated_num_nodes=int(getattr(args, "reward_dedicated_num_nodes", 0)),
            reward_dedicated_num_gpus_per_node=int(
                getattr(args, "reward_dedicated_num_gpus_per_node", 0)
            ),
            reward_placement_strategy=getattr(args, "reward_placement_strategy", "PACK"),
        )
