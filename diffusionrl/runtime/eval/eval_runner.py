"""Standalone evaluation runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from diffusionrl.reward.pipeline import score_from_rollout_outputs as compute_rewards_stage
from diffusionrl.runtime.pipeline.rollout_pipeline import (
    distributed_sample,
)
from diffusionrl.runtime.rollout.request_builder import RolloutRequestBuilder


class EvalRunner:
    """Own the evaluation sampling/reward path independently of rollout production."""

    def __init__(
        self,
        *,
        args: Any,
        sampling_config: Dict[str, Any],
        data_source: Any,
        reward_service: Any,
        algorithm: Any,
        default_prompt_batch_fn: Callable[[], Dict[str, Any]],
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.reward_service = reward_service
        self.algorithm = algorithm
        self.sampling_config = dict(sampling_config)
        self._default_prompt_batch_fn = default_prompt_batch_fn
        self._request_builder = RolloutRequestBuilder.from_args(
            args,
            sampling_defaults=self.sampling_config,
        )

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
        request = replace(
            request,
            sde_indices=None,
            decode_for_reward=True,
        )
        outputs = distributed_sample(
            actor_group=actor_group,
            request=request,
        )
        rewards, _ = compute_rewards_stage(
            reward_service=self.reward_service,
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
