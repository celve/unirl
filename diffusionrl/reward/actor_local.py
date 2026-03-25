"""Sampling-host reward precompute helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from diffusionrl.reward.schema import RewardSchema
from diffusionrl.types.sampling import RolloutSamples

from .pipeline import score_from_rollout_outputs
from .service import RewardService


class ActorLocalRewardPrecompute:
    """Shared adapter that precomputes rewards on rollout/training actors."""

    def __init__(
        self,
        reward_schema: RewardSchema,
        *,
        device_override: Optional[str] = None,
    ) -> None:
        if not isinstance(reward_schema, RewardSchema):
            raise TypeError(
                "ActorLocalRewardPrecompute requires RewardSchema, "
                f"got: {type(reward_schema).__name__}"
            )
        execution_plan = reward_schema.to_execution_plan()
        if not execution_plan.uses_sampling_actor_execution:
            raise ValueError(
                "ActorLocalRewardPrecompute requires reward_location='sampling_actor'."
            )
        if execution_plan.uses_ray_backend:
            raise ValueError(
                "ActorLocalRewardPrecompute does not support dedicated ray_pool reward backends. "
                "Use reward_location='driver' for reward_dedicated_* modes."
            )
        self.reward_schema = reward_schema
        if device_override is not None:
            from dataclasses import replace

            reward_schema = replace(
                reward_schema,
                local_reward_device=str(device_override),
            )
        self.executor = RewardService(reward_schema=reward_schema)

    def attach_to_output(
        self,
        *,
        output: RolloutSamples,
        prompts: List[str],
        prompt_ids: Optional[List[str]],
        sample_ids: Optional[List[str]],
        group_ids: Optional[List[str]],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
        keep_reward_media_for_driver: bool,
        samples_per_prompt: int,
    ) -> RolloutSamples:
        rewards, reward_components = score_from_rollout_outputs(
            reward_scoring_mode="service",
            reward_service=self.executor,
            samples_per_prompt=max(1, int(samples_per_prompt)),
            sampler_outputs=[output],
            prompts=list(prompts),
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )
        raw_meta = output.aux.get("metadata")
        meta = dict(raw_meta or {})
        meta["precomputed_rewards"] = [float(v) for v in rewards.tolist()]
        meta["precomputed_reward_components"] = {
            str(name): [float(v) for v in list(values or [])]
            for name, values in dict(reward_components or {}).items()
        }
        output.aux["metadata"] = meta
        if not keep_reward_media_for_driver:
            output.aux.pop("decoded_images", None)
            if isinstance(output.aux.get("metadata"), dict):
                output.aux["metadata"].pop("decoded_videos", None)
        return output

    def offload(self) -> None:
        self.executor.offload()

    def onload(self) -> None:
        self.executor.onload()

    def dispose(self) -> None:
        self.executor.dispose()

__all__ = [
    "ActorLocalRewardPrecompute",
]
