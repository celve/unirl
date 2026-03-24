"""Standalone evaluation runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from diffusionrl.orchestration.request_builder import RolloutRequestBuilder
from diffusionrl.orchestration.rollout_workflow import (
    distributed_sample,
)
from diffusionrl.reward.pipeline import score_from_rollout_outputs as compute_rewards_stage


class EvalRunner:
    """Own the evaluation sampling/reward path independently of rollout production."""

    def __init__(
        self,
        *,
        args: Any,
        sampling_config: Dict[str, Any],
        data_source: Any,
        reward_scoring_mode: str,
        reward_service: Any,
        algorithm: Any,
        default_prompt_batch_fn: Callable[[], Dict[str, Any]],
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.reward_scoring_mode = str(reward_scoring_mode)
        self.reward_service = reward_service
        self.algorithm = algorithm
        self.sampling_config = dict(sampling_config)
        self._default_prompt_batch_fn = default_prompt_batch_fn
        self._request_builder = RolloutRequestBuilder.from_args(
            args,
            sampling_defaults=self.sampling_config,
        )

    def _resolve_eval_request_overrides(self) -> Dict[str, Any]:
        rollout_eval = self.args.rollout.evaluation
        overrides: Dict[str, Any] = {}

        if getattr(rollout_eval, "num_inference_steps", None) is not None:
            overrides["num_inference_steps"] = int(rollout_eval.num_inference_steps)

        sampling_adapter = getattr(rollout_eval, "sampling_adapter", None)
        if sampling_adapter is not None and str(sampling_adapter).strip():
            overrides["sampling_adapter"] = str(sampling_adapter).strip()

        sampler_overrides: Dict[str, Any] = {}
        raw_sde_type = getattr(rollout_eval, "sde_type", None)
        if raw_sde_type is not None and str(raw_sde_type).strip():
            sampler_overrides["sde_type"] = str(raw_sde_type).strip()
        if getattr(rollout_eval, "eta", None) is not None:
            sampler_overrides["eta"] = float(rollout_eval.eta)
        if sampler_overrides:
            overrides["kwargs"] = {"sampler_overrides": sampler_overrides}

        return overrides

    def _build_eval_batch(self) -> Dict[str, Any]:
        rollout_eval = self.args.rollout.evaluation
        if self.data_source is not None and hasattr(self.data_source, "get_eval_samples"):
            eval_samples = self.data_source.get_eval_samples(rollout_eval.eval_batch_size)
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
        rollout_eval = self.args.rollout.evaluation
        prompts = list(eval_batch.get("prompts", [])[: rollout_eval.eval_batch_size])
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

        request_batch: Dict[str, Any] = {
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "metadata": prompt_metadata,
        }
        raw_request_kwargs = eval_batch.get("kwargs")
        if isinstance(raw_request_kwargs, dict):
            request_batch["kwargs"] = dict(raw_request_kwargs)

        eval_overrides = self._resolve_eval_request_overrides()
        if eval_overrides:
            request_batch.update(
                {key: value for key, value in eval_overrides.items() if key != "kwargs"}
            )
            override_kwargs = eval_overrides.get("kwargs")
            if isinstance(override_kwargs, dict):
                merged_kwargs = dict(request_batch.get("kwargs") or {})
                merged_sampler_overrides = dict(merged_kwargs.get("sampler_overrides") or {})
                merged_sampler_overrides.update(
                    dict(override_kwargs.get("sampler_overrides") or {})
                )
                if merged_sampler_overrides:
                    merged_kwargs["sampler_overrides"] = merged_sampler_overrides
                request_batch["kwargs"] = merged_kwargs

        request = self._request_builder.build_request(
            batch=request_batch,
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
            reward_scoring_mode=self.reward_scoring_mode,
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
