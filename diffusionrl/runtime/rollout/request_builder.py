"""Rollout request building and sub-batch execution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import partial
import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import tqdm as tqdm_

from diffusionrl.config.arguments import is_training_actor_sampling_mode
from diffusionrl.types.sampling import RolloutRequest

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledRequestResult:
    """Sampling result for one request split."""

    sampler_outputs: List[Any]


@dataclass(frozen=True)
class SampledRolloutBatch:
    """Aggregated outputs for one rollout after request-level sub-batching."""

    sampler_outputs: List[Any]
    train_prompts: List[str]
    train_prompt_ids: Optional[List[str]]
    sample_ids: Optional[List[str]]
    group_ids: Optional[List[str]]
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]]


class RolloutRequestBuilder:
    """Build typed rollout requests and execute sample-level request splits."""

    def __init__(
        self,
        *,
        is_direct_sampling_mode: bool,
        max_samples_per_request: Optional[int],
        sampling_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.is_direct_sampling_mode = bool(is_direct_sampling_mode)
        self.max_samples_per_request = (
            None if max_samples_per_request is None else int(max_samples_per_request)
        )
        self.sampling_defaults = dict(sampling_defaults or {})

    @staticmethod
    def _resolve_explicit_group_ids(raw_group_ids: Any, *, base_size: int) -> Optional[List[str]]:
        if not isinstance(raw_group_ids, list) or len(raw_group_ids) != base_size:
            return None

        resolved: List[str] = []
        for prompt_idx, raw_group_id in enumerate(raw_group_ids):
            group_id = str(raw_group_id).strip()
            if not group_id:
                raise ValueError(
                    "Explicit batch['group_ids'] must provide a non-empty value for every prompt. "
                    f"Found invalid group_id at prompt_idx={prompt_idx}."
                )
            resolved.append(group_id)
        return resolved

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        sampling_defaults: Dict[str, Any],
    ) -> "RolloutRequestBuilder":
        return cls(
            is_direct_sampling_mode=is_training_actor_sampling_mode(args),
            max_samples_per_request=getattr(args.sampling, "max_samples_per_request", None),
            sampling_defaults=sampling_defaults,
        )

    def build_request(
        self,
        *,
        batch: Dict[str, Any],
        samples_per_prompt: int,
    ) -> RolloutRequest:
        k = int(samples_per_prompt)
        if k < 1:
            raise ValueError(
                f"samples_per_prompt must be >= 1 when building rollout requests, got {k}."
            )
        prompts = batch.get("prompts") or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Rollout request building requires non-empty batch['prompts'].")

        base_size = len(prompts)
        expanded_prompts = [prompt for prompt in prompts for _ in range(k)]

        raw_prompt_ids = batch.get("prompt_ids")
        if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == base_size:
            base_prompt_ids = [str(prompt_id) for prompt_id in raw_prompt_ids]
        else:
            base_prompt_ids = [f"prompt:{prompt_idx}" for prompt_idx in range(base_size)]
        expanded_prompt_ids = [prompt_id for prompt_id in base_prompt_ids for _ in range(k)]

        raw_group_ids = batch.get("group_ids")
        resolved_group_ids = self._resolve_explicit_group_ids(
            raw_group_ids,
            base_size=base_size,
        )
        if resolved_group_ids is not None:
            base_group_ids = resolved_group_ids
        else:
            base_group_ids = list(base_prompt_ids)
        expanded_group_ids = [group_id for group_id in base_group_ids for _ in range(k)]

        raw_noise_group_ids = batch.get("noise_group_ids")
        if isinstance(raw_noise_group_ids, list) and len(raw_noise_group_ids) == base_size:
            base_noise_group_ids = [str(group_id) for group_id in raw_noise_group_ids]
        else:
            base_noise_group_ids = list(base_prompt_ids)
        expanded_noise_group_ids = [group_id for group_id in base_noise_group_ids for _ in range(k)]

        sample_ids = [
            f"prompt:{base_prompt_ids[prompt_idx]}:sample:{replica_idx}"
            for prompt_idx in range(base_size)
            for replica_idx in range(k)
        ]

        prompt_metadata = batch.get("metadata")
        if isinstance(prompt_metadata, list) and len(prompt_metadata) == base_size:
            expanded_prompt_metadata = [item for item in prompt_metadata for _ in range(k)]
        else:
            expanded_prompt_metadata = None

        latents = batch.get("latents")
        if torch.is_tensor(latents) and latents.shape[0] == base_size:
            latents = latents.repeat_interleave(k, dim=0)

        raw_num_inference_steps = batch.get(
            "num_inference_steps",
            self.sampling_defaults.get("num_inference_steps"),
        )
        raw_guidance_scale = batch.get(
            "guidance_scale",
            self.sampling_defaults.get("guidance_scale"),
        )
        raw_height = batch.get("height", self.sampling_defaults.get("height"))
        raw_width = batch.get("width", self.sampling_defaults.get("width"))
        raw_num_frames = batch.get("num_frames", self.sampling_defaults.get("num_frames"))
        if raw_num_inference_steps is None:
            raise ValueError("Rollout request building requires resolved num_inference_steps.")
        if raw_guidance_scale is None:
            raise ValueError("Rollout request building requires resolved guidance_scale.")
        if raw_height is None or raw_width is None or raw_num_frames is None:
            raise ValueError(
                "Rollout request building requires resolved geometry "
                f"(height={raw_height}, width={raw_width}, num_frames={raw_num_frames})."
            )
        num_inference_steps = int(raw_num_inference_steps)
        guidance_scale = float(raw_guidance_scale)
        height = int(raw_height)
        width = int(raw_width)
        num_frames = int(raw_num_frames)
        sampling_adapter = batch.get(
            "sampling_adapter",
            self.sampling_defaults.get("sampling_adapter"),
        )
        init_same_noise = bool(
            batch.get(
                "init_same_noise",
                self.sampling_defaults.get("init_same_noise", False),
            )
        )
        raw_seed = batch.get("seed", self.sampling_defaults.get("seed"))
        seed = None if raw_seed is None else int(raw_seed)

        return RolloutRequest(
            prompts=expanded_prompts,
            prompt_ids=expanded_prompt_ids,
            sample_ids=sample_ids,
            group_ids=expanded_group_ids,
            noise_group_ids=expanded_noise_group_ids,
            prompt_metadata=expanded_prompt_metadata,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            latents=latents,
            init_same_noise=init_same_noise,
            samples_per_prompt=k,
            sampling_adapter=sampling_adapter,
        )

    def estimate_request_batches(
        self,
        *,
        prompt_count: int,
        samples_per_prompt: int,
    ) -> int:
        resolved_prompt_count = int(prompt_count)
        resolved_samples_per_prompt = int(samples_per_prompt)
        if resolved_prompt_count <= 0:
            return 1
        if resolved_samples_per_prompt < 1:
            raise ValueError(
                "samples_per_prompt must be >= 1 for request planning. "
                f"Got {resolved_samples_per_prompt}."
            )
        total_samples = resolved_prompt_count * resolved_samples_per_prompt
        if (
            not self.is_direct_sampling_mode
            or self.max_samples_per_request is None
            or self.max_samples_per_request >= total_samples
        ):
            return 1
        return (total_samples + self.max_samples_per_request - 1) // self.max_samples_per_request

    def build_request_batches(
        self,
        *,
        batch: Dict[str, Any],
        samples_per_prompt: int,
    ) -> List[Tuple[int, RolloutRequest]]:
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Rollout request building requires non-empty batch['prompts'].")

        requests_per_rollout = self.estimate_request_batches(
            prompt_count=len(prompts),
            samples_per_prompt=samples_per_prompt,
        )
        if requests_per_rollout <= 1:
            return [
                (
                    0,
                    self.build_request(
                        batch=batch,
                        samples_per_prompt=samples_per_prompt,
                    ),
                )
            ]

        full_request = self.build_request(
            batch=batch,
            samples_per_prompt=samples_per_prompt,
        )
        request_batch_size = len(full_request.prompts)
        if self.max_samples_per_request is None or self.max_samples_per_request < 1:
            raise ValueError("sampling.max_samples_per_request must be >= 1.")
        requests: List[Tuple[int, RolloutRequest]] = []
        for batch_index, start in enumerate(range(0, request_batch_size, self.max_samples_per_request)):
            end = min(start + self.max_samples_per_request, request_batch_size)
            requests.append(
                (
                    batch_index,
                    full_request.slice_prompts(start, end),
                )
            )
        return requests

    def execute_request_batches(
        self,
        *,
        request_batches: Optional[List[Tuple[int, RolloutRequest]]],
        sample_request: Callable[[RolloutRequest], SampledRequestResult],
        validate_sampler_outputs: Optional[Callable[[List[Any]], None]] = None,
        rollout_id: Optional[int] = None,
    ) -> SampledRolloutBatch:
        """Execute one rollout, optionally split into multiple typed requests."""

        planned_requests = list(request_batches or [])
        if not planned_requests:
            raise ValueError("Rollout request builder produced no sampling requests.")

        all_outputs: List[Any] = []
        all_train_prompts: List[str] = []
        all_train_prompt_ids: List[str] = []
        all_sample_ids: List[str] = []
        all_group_ids: List[str] = []
        all_prompt_metadata: List[Optional[Dict[str, Any]]] = []
        requests_per_rollout = len(planned_requests)
        rank = int(os.environ.get("RANK", 0))
        rollout_seed_offset = 0 if rollout_id is None else int(rollout_id)

        for batch_idx, current_request in tqdm(
            planned_requests,
            desc=f"Rollout {rollout_id} on rank {rank}",
            unit="request",
            disable=(rank != 0),
        ):
            if not isinstance(current_request.prompts, list) or len(current_request.prompts) == 0:
                raise ValueError("Rollout request builder requires non-empty request.prompts.")
            if current_request.seed is not None:
                current_request = replace(
                    current_request,
                    seed=int(current_request.seed) + rollout_seed_offset,
                )

            if requests_per_rollout > 1:
                logger.debug(
                    "rollout=%s direct-sampling request %s/%s samples=%s",
                    rollout_id,
                    batch_idx + 1,
                    requests_per_rollout,
                    len(current_request.prompts),
                )

            _req_t0 = _time.perf_counter()
            sampled_request = sample_request(current_request)
            _req_t1 = _time.perf_counter()
            sampler_outputs = sampled_request.sampler_outputs

            if validate_sampler_outputs is not None:
                validate_sampler_outputs(sampler_outputs)
            _req_t2 = _time.perf_counter()
            logger.debug(
                "[TIMING] sample_rollout req=%d/%d: sample=%.2fs validate=%.2fs",
                batch_idx + 1,
                requests_per_rollout,
                _req_t1 - _req_t0,
                _req_t2 - _req_t1,
            )

            all_outputs.extend(sampler_outputs)
            all_train_prompts.extend(list(current_request.prompts))
            if current_request.prompt_ids is not None:
                all_train_prompt_ids.extend(list(current_request.prompt_ids))
            if current_request.sample_ids is not None:
                all_sample_ids.extend(list(current_request.sample_ids))
            if current_request.group_ids is not None:
                all_group_ids.extend(list(current_request.group_ids))
            if current_request.prompt_metadata is not None:
                all_prompt_metadata.extend(list(current_request.prompt_metadata))
            else:
                all_prompt_metadata.extend([None] * len(current_request.prompts))

        return SampledRolloutBatch(
            sampler_outputs=all_outputs,
            train_prompts=all_train_prompts,
            train_prompt_ids=all_train_prompt_ids if all_train_prompt_ids else None,
            sample_ids=all_sample_ids if all_sample_ids else None,
            group_ids=all_group_ids if all_group_ids else None,
            prompt_metadata=all_prompt_metadata if any(item is not None for item in all_prompt_metadata) else None,
        )


__all__ = [
    "RolloutRequestBuilder",
    "SampledRequestResult",
    "SampledRolloutBatch",
]
