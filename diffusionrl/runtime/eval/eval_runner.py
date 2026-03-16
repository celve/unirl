"""Standalone evaluation runner."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from diffusionrl.config.build_domain_args import RewardSchema
from diffusionrl.runtime.pipeline.rollout_pipeline import (
    compute_rewards as compute_rewards_stage,
    distributed_sample,
)
from diffusionrl.runtime.rollout.request_builder import RolloutRequestBuilder
from diffusionrl.types.sampling import RolloutRequest


class EvalRunner:
    """Own the evaluation sampling/reward path independently of rollout production."""

    def __init__(
        self,
        *,
        args: Any,
        data_source: Any,
        reward_schema: RewardSchema,
        reward_service: Any,
        algorithm: Any,
        default_prompt_batch_fn: Callable[[], Dict[str, Any]],
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.reward_schema = reward_schema
        self.reward_service = reward_service
        self.algorithm = algorithm
        self._default_prompt_batch_fn = default_prompt_batch_fn
        self._request_builder = RolloutRequestBuilder.from_args(args)

    def _build_eval_batch(self) -> Dict[str, Any]:
        if self.data_source is not None and hasattr(self.data_source, "get_eval_samples"):
            eval_samples = self.data_source.get_eval_samples(self.args.rollout.eval_batch_size)
            if isinstance(eval_samples, dict):
                return dict(eval_samples)
            if isinstance(eval_samples, list):
                return {"prompts": list(eval_samples)}
            raise TypeError(
                "DataSource.get_eval_samples() must return List[str] or Dict[str, Any] "
                f"with at least 'prompts'. Got {type(eval_samples).__name__}."
            )
        return dict(self._default_prompt_batch_fn())

    def evaluate(
        self,
        *,
        rollout_id: int,
        actor_group: Any,
    ) -> Dict[str, Any]:
        eval_batch = self._build_eval_batch()
        prompts = list(eval_batch.get("prompts", [])[: self.args.rollout.eval_batch_size])
        prompt_ids = eval_batch.get("prompt_ids")
        if isinstance(prompt_ids, list):
            prompt_ids = prompt_ids[: len(prompts)]
        else:
            prompt_ids = None
        prompt_metadata = eval_batch.get("metadata")
        if isinstance(prompt_metadata, list):
            prompt_metadata = prompt_metadata[: len(prompts)]
        else:
            prompt_metadata = None

        request = self._request_builder.build_request(
            batch={
                "prompts": prompts,
                "prompt_ids": prompt_ids,
                "metadata": prompt_metadata,
            },
            samples_per_prompt=int(getattr(self.algorithm, "samples_per_prompt", 1)),
        )
        request = RolloutRequest(
            prompts=request.prompts,
            prompt_ids=request.prompt_ids,
            sample_ids=request.sample_ids,
            group_ids=request.group_ids,
            noise_group_ids=request.noise_group_ids,
            prompt_metadata=request.prompt_metadata,
            num_inference_steps=int(self.args.sampling.num_inference_steps),
            guidance_scale=float(self.args.sampling.guidance_scale),
            height=int(self.args.height),
            width=int(self.args.width),
            num_frames=int(self.args.num_frames),
            init_same_noise=bool(getattr(self.args.sampling, "init_same_noise", False)),
            samples_per_prompt=int(getattr(self.algorithm, "samples_per_prompt", 1)),
            sde_indices=None,
            decode_for_reward=True,
        )
        outputs = distributed_sample(
            actor_group=actor_group,
            request=request,
        )
        rewards, _ = compute_rewards_stage(
            reward_service=self.reward_service,
            reward_path=str(self.reward_schema.reward_path or ""),
            samples_per_prompt=int(getattr(self.algorithm, "samples_per_prompt", 1)),
            sampler_outputs=outputs,
            prompts=prompts,
            prompt_ids=prompt_ids,
            prompt_metadata=prompt_metadata,
        )
        return {
            "rollout_id": int(rollout_id),
            "num_samples": len(prompts),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "prompts": prompts,
        }
