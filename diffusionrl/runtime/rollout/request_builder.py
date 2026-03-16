"""Rollout request building and sub-batch execution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    """Build typed rollout requests and execute prompt-aligned sub-batches."""

    def __init__(
        self,
        *,
        is_direct_sampling_mode: bool,
        max_samples_per_request: Optional[int],
    ) -> None:
        self.is_direct_sampling_mode = bool(is_direct_sampling_mode)
        self.max_samples_per_request = (
            None if max_samples_per_request is None else max(1, int(max_samples_per_request))
        )

    @classmethod
    def from_args(cls, args: Any) -> "RolloutRequestBuilder":
        return cls(
            is_direct_sampling_mode=is_training_actor_sampling_mode(args),
            max_samples_per_request=getattr(args.sampling, "max_samples_per_request", None),
        )

    @staticmethod
    def _slice_prompt_aligned_batch(
        batch: Dict[str, Any],
        *,
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Rollout sampling requires non-empty batch['prompts'].")

        sliced: Dict[str, Any] = {}
        prompt_count = len(prompts)
        for key, value in batch.items():
            if key == "prompts":
                continue
            if isinstance(value, list) and len(value) == prompt_count:
                sliced[key] = value[start:end]
                continue
            try:
                if torch.is_tensor(value) and value.shape[0] == prompt_count:
                    sliced[key] = value[start:end]
                    continue
            except Exception:
                pass
            sliced[key] = value

        sliced["prompts"] = list(prompts[start:end])
        return sliced

    @staticmethod
    def build_request(
        *,
        batch: Dict[str, Any],
        samples_per_prompt: int,
    ) -> RolloutRequest:
        k = max(1, int(samples_per_prompt))
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
        if isinstance(raw_group_ids, list) and len(raw_group_ids) == base_size:
            base_group_ids = [str(group_id) for group_id in raw_group_ids]
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

        return RolloutRequest(
            prompts=expanded_prompts,
            prompt_ids=expanded_prompt_ids,
            sample_ids=sample_ids,
            group_ids=expanded_group_ids,
            noise_group_ids=expanded_noise_group_ids,
            prompt_metadata=expanded_prompt_metadata,
            latents=latents,
            samples_per_prompt=k,
        )

    def estimate_request_batches(
        self,
        *,
        prompt_count: int,
        samples_per_prompt: int,
    ) -> int:
        prompt_count = max(0, int(prompt_count))
        samples_per_prompt = max(1, int(samples_per_prompt))
        total_samples = prompt_count * samples_per_prompt
        if (
            prompt_count <= 0
            or not self.is_direct_sampling_mode
            or self.max_samples_per_request is None
            or self.max_samples_per_request >= total_samples
        ):
            return 1
        if self.max_samples_per_request % samples_per_prompt != 0:
            raise ValueError(
                "sampling.max_samples_per_request must be divisible by algorithm.samples_per_prompt. "
                f"Got max_samples_per_request={self.max_samples_per_request}, "
                f"samples_per_prompt={samples_per_prompt}."
            )
        prompts_per_request = self.max_samples_per_request // samples_per_prompt
        if prompts_per_request < 1 or prompt_count % prompts_per_request != 0:
            raise ValueError(
                "Prompt batch size must be divisible by prompts_per_request derived from "
                "sampling.max_samples_per_request. "
                f"Got prompt_count={prompt_count}, prompts_per_request={prompts_per_request}."
            )
        return prompt_count // prompts_per_request

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

        prompts_per_request = self.max_samples_per_request // max(1, int(samples_per_prompt))
        requests: List[Tuple[int, RolloutRequest]] = []
        batch_index = 0
        for start in range(0, len(prompts), prompts_per_request):
            end = start + prompts_per_request
            request_batch = self._slice_prompt_aligned_batch(batch, start=start, end=end)
            requests.append(
                (
                    batch_index,
                    self.build_request(
                        batch=request_batch,
                        samples_per_prompt=samples_per_prompt,
                    ),
                )
            )
            batch_index += 1
        return requests

    def execute_request_batches(
        self,
        *,
        request_batches: Optional[List[Tuple[int, RolloutRequest]]],
        sample_request: Callable[[RolloutRequest], SampledRequestResult],
        attach_embeddings: Optional[Callable[[List[Any], List[str]], None]] = None,
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

        for batch_idx, current_request in tqdm(
            planned_requests,
            desc=f"Rollout {rollout_id} on rank {rank}",
            unit="request",
            disable=(rank != 0),
        ):
            if not isinstance(current_request.prompts, list) or len(current_request.prompts) == 0:
                raise ValueError("Rollout request builder requires non-empty request.prompts.")

            if requests_per_rollout > 1:
                logger.debug(
                    "rollout=%s direct-sampling request %s/%s prompts=%s",
                    rollout_id,
                    batch_idx + 1,
                    requests_per_rollout,
                    len(current_request.prompts),
                )

            _req_t0 = _time.perf_counter()
            sampled_request = sample_request(current_request)
            _req_t1 = _time.perf_counter()
            sampler_outputs = sampled_request.sampler_outputs

            if attach_embeddings is not None:
                attach_embeddings(sampler_outputs, list(current_request.prompts))
            _req_t2 = _time.perf_counter()

            if validate_sampler_outputs is not None:
                validate_sampler_outputs(sampler_outputs)
            _req_t3 = _time.perf_counter()
            logger.warning(
                "[TIMING] sample_rollout req=%d/%d: sample=%.2fs embed=%.2fs validate=%.2fs",
                batch_idx + 1,
                requests_per_rollout,
                _req_t1 - _req_t0,
                _req_t2 - _req_t1,
                _req_t3 - _req_t2,
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
