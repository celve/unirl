"""Stable rollout-side service interface exposed to rollout functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from diffusionrl.rollout.primitives import (
    build_rollout_request,
    compute_advantages as compute_advantages_stage,
    distributed_sample,
    normalize_rollout_outputs,
    plan_request_batches,
)
from diffusionrl.reward.pipeline import score_from_rollout_outputs as score_reward_stage
from diffusionrl.types.sampling import RolloutRequest
from diffusionrl.types.training_batch import TrainingBatch


@dataclass(frozen=True)
class LaunchedSamplingRequest:
    """One launched sampling request plus its future handle."""

    request: RolloutRequest
    future: Any


def load_prompt_batch_from_source(
    *,
    data_source: Any,
    prompt_batch_size: int,
) -> Dict[str, Any]:
    """Fetch one rollout prompt batch from the configured data source."""
    if data_source is None:
        raise RuntimeError("Rollout services require an initialized data source.")
    samples = data_source.get_samples(int(prompt_batch_size))
    if isinstance(samples, dict):
        return samples
    raise TypeError(
        "DataSource.get_samples() must return Dict[str, Any] with at least 'prompts'. "
        f"Got {type(samples).__name__}."
    )


def build_eval_request_batch(
    *,
    data_source: Any,
    prompt_batch_size: int,
    evaluation_settings: Any,
) -> Dict[str, Any]:
    """Build the canonical evaluation request payload before request expansion."""
    eval_batch_size = int(getattr(evaluation_settings, "eval_batch_size", 0) or 0)
    if data_source is not None and hasattr(data_source, "get_eval_samples"):
        eval_samples = data_source.get_eval_samples(eval_batch_size)
        if isinstance(eval_samples, dict):
            eval_batch = dict(eval_samples)
        elif isinstance(eval_samples, list):
            eval_batch = {"prompts": list(eval_samples)}
        else:
            raise TypeError(
                "DataSource.get_eval_samples() must return List[str] or Dict[str, Any] "
                f"with at least 'prompts'. Got {type(eval_samples).__name__}."
            )
    else:
        eval_batch = dict(
            load_prompt_batch_from_source(
                data_source=data_source,
                prompt_batch_size=prompt_batch_size,
            )
        )

    prompts = list(eval_batch.get("prompts", [])[:eval_batch_size])
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

    eval_overrides: Dict[str, Any] = {}
    if getattr(evaluation_settings, "num_inference_steps", None) is not None:
        eval_overrides["num_inference_steps"] = int(evaluation_settings.num_inference_steps)

    sampling_adapter = getattr(evaluation_settings, "sampling_adapter", None)
    if sampling_adapter is not None and str(sampling_adapter).strip():
        eval_overrides["sampling_adapter"] = str(sampling_adapter).strip()

    sampler_overrides: Dict[str, Any] = {}
    raw_sde_type = getattr(evaluation_settings, "sde_type", None)
    if raw_sde_type is not None and str(raw_sde_type).strip():
        sampler_overrides["sde_type"] = str(raw_sde_type).strip()
    if getattr(evaluation_settings, "eta", None) is not None:
        sampler_overrides["eta"] = float(evaluation_settings.eta)
    if sampler_overrides:
        eval_overrides["kwargs"] = {"sampler_overrides": sampler_overrides}

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
    return request_batch


def compute_dataset_step_info(
    *,
    data_source: Any,
    prompts_per_rollout: int,
) -> Dict[str, Any]:
    """Compute rollout-step progress information for the current dataset."""
    drop_last = bool(getattr(data_source, "drop_last", False))
    info: Dict[str, Any] = {
        "num_prompts": 0,
        "prompts_per_rollout": int(prompts_per_rollout),
        "estimated_steps_per_dataset_pass": 0,
        "steps_before_reset": 0,
        "remainder_prompts": 0,
        "drop_last": drop_last,
        "exact_dataset_pass_per_cycle": False,
    }

    if data_source is None or not hasattr(data_source, "num_prompts"):
        return info

    num_prompts = int(data_source.num_prompts)
    info["num_prompts"] = num_prompts
    if num_prompts <= 0:
        return info

    estimated_steps = (num_prompts + prompts_per_rollout - 1) // prompts_per_rollout
    remainder = num_prompts % prompts_per_rollout
    if drop_last:
        steps_before_reset = num_prompts // prompts_per_rollout
    else:
        steps_before_reset = estimated_steps

    info.update(
        {
            "estimated_steps_per_dataset_pass": int(estimated_steps),
            "steps_before_reset": int(steps_before_reset),
            "remainder_prompts": int(remainder),
            "exact_dataset_pass_per_cycle": bool(
                remainder == 0 and steps_before_reset == estimated_steps
            ),
        }
    )
    return info


class RolloutServices:
    """Thin runtime handle for data loading, request execution, reward, and assembly."""

    def __init__(
        self,
        *,
        algorithm: Any,
        data_source: Any,
        reward_scoring_mode: str,
        reward_service: Any,
        is_direct_sampling_mode: bool,
        max_samples_per_request: Optional[int],
        reward_component_weights: Optional[Dict[str, float]],
        prompt_batch_size: int,
        evaluation_settings: Any,
        sampler_validation_config: Optional[Dict[str, Any]],
        sampling_config: Dict[str, Any],
        sampling_requirements: Any,
        debug_mode: str,
        debug_output_dir: Optional[str],
    ) -> None:
        self.algorithm = algorithm
        self.data_source = data_source
        self.reward_scoring_mode = str(reward_scoring_mode)
        self.reward_service = reward_service
        self.is_direct_sampling_mode = bool(is_direct_sampling_mode)
        self.max_samples_per_request = (
            None if max_samples_per_request is None else int(max_samples_per_request)
        )
        self.reward_component_weights = dict(reward_component_weights or {})
        self.prompt_batch_size = int(prompt_batch_size)
        self.evaluation_settings = evaluation_settings
        self.sampler_validation_config = dict(sampler_validation_config or {})
        self.sampling_config = dict(sampling_config)
        self.sampling_requirements = sampling_requirements
        self.debug_mode = str(debug_mode or "none")
        self.debug_output_dir = (
            None
            if debug_output_dir is None or not str(debug_output_dir).strip()
            else str(debug_output_dir)
        )
        self._sampling_group = None

    @property
    def samples_per_prompt(self) -> int:
        return int(getattr(self.algorithm, "samples_per_prompt", 1))

    def attach_sampling_group(self, actor_group: Any) -> None:
        self._sampling_group = actor_group

    def get_sampling_group(self) -> Any:
        actor_group = self._sampling_group
        if actor_group is None:
            raise RuntimeError("No sampling group attached. Call attach_sampling_group() first.")
        return actor_group

    def dispose(self) -> None:
        if self.reward_service is not None:
            self.reward_service.dispose()
            self.reward_service = None
        self._sampling_group = None

    def load_prompt_batch(self) -> Dict[str, Any]:
        return load_prompt_batch_from_source(
            data_source=self.data_source,
            prompt_batch_size=self.prompt_batch_size,
        )

    def build_request(
        self,
        *,
        batch: Dict[str, Any],
        samples_per_prompt: Optional[int] = None,
    ) -> RolloutRequest:
        return build_rollout_request(
            batch=batch,
            samples_per_prompt=(
                self.samples_per_prompt if samples_per_prompt is None else int(samples_per_prompt)
            ),
            sampling_defaults=self.sampling_config,
        )

    def plan_request_batches(
        self,
        *,
        batch: Dict[str, Any],
        samples_per_prompt: Optional[int] = None,
    ) -> List[Tuple[int, RolloutRequest]]:
        return plan_request_batches(
            batch=batch,
            samples_per_prompt=(
                self.samples_per_prompt if samples_per_prompt is None else int(samples_per_prompt)
            ),
            is_direct_sampling_mode=self.is_direct_sampling_mode,
            max_samples_per_request=self.max_samples_per_request,
            sampling_defaults=self.sampling_config,
        )

    def execute_sampling_request(
        self,
        *,
        request: RolloutRequest,
        actor_group: Any = None,
        sde_indices: Optional[Set[int]] = None,
        requirements: Any = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Run distributed sampling for one resolved rollout request."""
        typed_request = self.prepare_sampling_request(
            request=request,
            sde_indices=sde_indices,
            requirements=requirements,
            sampling_overrides=sampling_overrides,
        )
        return distributed_sample(
            actor_group=self.get_sampling_group() if actor_group is None else actor_group,
            request=typed_request,
        )

    def prepare_sampling_request(
        self,
        *,
        request: RolloutRequest,
        sde_indices: Optional[Set[int]] = None,
        requirements: Any = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> RolloutRequest:
        """Resolve all sampling-time fields on a request before actor dispatch."""
        prompts = request.prompts or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "Rollout sampling requires non-empty text prompts in request.prompts. "
                "Prompt-embedding-only batches are no longer supported."
            )

        overrides = dict(sampling_overrides or {})
        if self.debug_output_dir:
            overrides.setdefault("debug_output_dir", self.debug_output_dir)

        resolved_num_inference_steps = overrides.pop("num_inference_steps", request.num_inference_steps)
        resolved_guidance_scale = overrides.pop("guidance_scale", request.guidance_scale)
        resolved_height = overrides.pop("height", request.height)
        resolved_width = overrides.pop("width", request.width)
        resolved_num_frames = overrides.pop("num_frames", request.num_frames)
        if resolved_num_inference_steps is None:
            raise ValueError("RolloutRequest.num_inference_steps must be resolved before sampling.")
        if resolved_guidance_scale is None:
            raise ValueError("RolloutRequest.guidance_scale must be resolved before sampling.")
        if resolved_height is None or resolved_width is None or resolved_num_frames is None:
            raise ValueError(
                "RolloutRequest geometry must be resolved before sampling "
                f"(height={resolved_height}, width={resolved_width}, num_frames={resolved_num_frames})."
            )

        resolved_requirements = requirements or self.sampling_requirements
        requires_trajectory = True
        requires_log_prob = True
        if resolved_requirements is not None:
            requires_trajectory = bool(getattr(resolved_requirements, "requires_trajectory", True))
            requires_log_prob = bool(getattr(resolved_requirements, "requires_log_prob", True))

        merged_sampling = dict(request.sampling)
        merged_sampling["sde_indices"] = sde_indices
        merged_sampling["decode_for_reward"] = True
        merged_sampling["keep_reward_media_for_driver"] = bool(
            overrides.pop("_keep_reward_media_for_driver", False)
        )
        merged_sampling["init_same_noise"] = bool(request.sampling.get("init_same_noise", False))
        merged_sampling["samples_per_prompt"] = max(
            1, int(request.sampling.get("samples_per_prompt", 1))
        )
        merged_sampling["return_trajectories"] = requires_trajectory
        merged_sampling["return_log_probs"] = requires_log_prob
        merged_kwargs = dict(request.sampling.get("kwargs") or {})
        merged_kwargs.update(overrides)
        merged_sampling["kwargs"] = merged_kwargs

        return RolloutRequest(
            prompts=list(request.prompts),
            num_inference_steps=int(resolved_num_inference_steps),
            guidance_scale=float(resolved_guidance_scale),
            height=int(resolved_height),
            width=int(resolved_width),
            num_frames=int(resolved_num_frames),
            sampling=merged_sampling,
            meta=request.meta,
            inputs=request.inputs,
        )

    def launch_sampling_request(
        self,
        *,
        request: RolloutRequest,
        actor_group: Any = None,
        sde_indices: Optional[Set[int]] = None,
        requirements: Any = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> LaunchedSamplingRequest:
        """Launch sampling for one request without blocking on completion."""
        resolved_actor_group = self.get_sampling_group() if actor_group is None else actor_group
        typed_request = self.prepare_sampling_request(
            request=request,
            sde_indices=sde_indices,
            requirements=requirements,
            sampling_overrides=sampling_overrides,
        )
        future = resolved_actor_group.async_generate(typed_request)
        return LaunchedSamplingRequest(
            request=typed_request,
            future=future,
        )

    def resolve_launched_sampling_request(
        self,
        *,
        launched_request: LaunchedSamplingRequest,
    ) -> List[Any]:
        """Resolve one previously launched sampling request."""
        import ray

        raw_outputs = ray.get(launched_request.future)
        return normalize_rollout_outputs(raw_outputs)

    def score_rewards(
        self,
        *,
        request: RolloutRequest,
        sampler_outputs: List[Any],
        samples_per_prompt_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        """Run the default reward stage for sampled rollout outputs."""
        samples_per_prompt = int(
            self.samples_per_prompt
            if samples_per_prompt_override is None
            else samples_per_prompt_override
        )
        return score_reward_stage(
            reward_scoring_mode=self.reward_scoring_mode,
            reward_service=self.reward_service,
            samples_per_prompt=samples_per_prompt,
            sampler_outputs=sampler_outputs,
            prompts=list(request.prompts),
            prompt_ids=request.meta.get("prompt_ids"),
            sample_ids=request.meta.get("sample_ids"),
            group_ids=request.meta.get("group_ids"),
            prompt_metadata=request.meta.get("prompt_metadata"),
        )

    def compute_advantages(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        component_rewards: Optional[Dict[str, List[float]]] = None,
        reward_components: Optional[Dict[str, List[float]]] = None,
    ) -> torch.Tensor:
        if component_rewards is not None and reward_components is not None:
            raise ValueError(
                "compute_advantages accepts either component_rewards or reward_components, not both."
            )
        resolved_component_rewards = (
            component_rewards if component_rewards is not None else reward_components
        )
        return compute_advantages_stage(
            algorithm=self.algorithm,
            rewards=rewards,
            group_ids=group_ids,
            component_rewards=resolved_component_rewards,
            reward_component_weights=self.reward_component_weights,
        )

    def assemble_training_batch(
        self,
        *,
        request: RolloutRequest,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> TrainingBatch:
        return self.algorithm.assemble_training_batch(
            request=request,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            sde_indices=sde_indices,
        )


__all__ = [
    "LaunchedSamplingRequest",
    "RolloutServices",
    "build_eval_request_batch",
    "compute_dataset_step_info",
    "load_prompt_batch_from_source",
]
