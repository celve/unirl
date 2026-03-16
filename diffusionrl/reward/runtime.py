"""Reward runtime helpers that separate semantics from deployment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from diffusionrl.config.build_domain_args import RewardSchema
from diffusionrl.runtime.pipeline.rollout_pipeline import compute_rewards as compute_rollout_rewards
from diffusionrl.types.sampling import RolloutOutput

from .service import LocalRewardExecutor, RewardService


def create_manager_reward_service(
    reward_schema: RewardSchema,
    *,
    reward_pg_result: Optional[Any] = None,
) -> Optional[RewardService]:
    """Create the manager-side reward executor when manager owns reward runtime."""
    if reward_schema.to_execution_plan().uses_sampling_actor_execution:
        return None
    return RewardService(
        reward_schema=reward_schema,
        reward_pg_result=reward_pg_result,
    )


class SamplingActorRewardRuntime:
    """Actor-local reward runtime shared by rollout and training actors."""

    def __init__(
        self,
        reward_schema: RewardSchema,
        *,
        device_override: Optional[str] = None,
    ) -> None:
        if not isinstance(reward_schema, RewardSchema):
            raise TypeError(
                "SamplingActorRewardRuntime requires RewardSchema, "
                f"got: {type(reward_schema).__name__}"
            )
        execution_plan = reward_schema.to_execution_plan()
        if not execution_plan.uses_sampling_actor_execution:
            raise ValueError(
                "SamplingActorRewardRuntime requires reward_location='sampling_actor'."
            )
        self.reward_schema = reward_schema
        self.reward_spec = reward_schema.to_spec()
        self.execution_plan = execution_plan
        self.executor = LocalRewardExecutor(
            reward_schema=reward_schema,
            device_override=device_override,
        )

    def attach_to_output(
        self,
        *,
        output: RolloutOutput,
        prompts: List[str],
        prompt_ids: Optional[List[str]],
        sample_ids: Optional[List[str]],
        group_ids: Optional[List[str]],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
        keep_reward_media_for_manager: bool,
        samples_per_prompt: int,
    ) -> RolloutOutput:
        rewards, reward_components = compute_rollout_rewards(
            reward_service=self.executor,
            reward_path=str(self.reward_spec.reward_path or ""),
            samples_per_prompt=max(1, int(samples_per_prompt)),
            sampler_outputs=[output],
            prompts=list(prompts),
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )
        meta = dict(output.metadata or {})
        meta["precomputed_rewards"] = [float(v) for v in rewards.tolist()]
        meta["precomputed_reward_components"] = {
            str(name): [float(v) for v in list(values or [])]
            for name, values in dict(reward_components or {}).items()
        }
        output.metadata = meta
        if not keep_reward_media_for_manager:
            output.decoded_images = None
            if isinstance(output.metadata, dict):
                output.metadata.pop("decoded_videos", None)
        return output

    def offload(self) -> None:
        self.executor.offload()

    def onload(self) -> None:
        self.executor.onload()

    def dispose(self) -> None:
        self.executor.dispose()
