"""Low-level rollout planning, sampling, validation, and aggregation helpers."""

from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import tqdm as tqdm_

from diffusionrl.types.sampling import RolloutRequest, RolloutSamples

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = logging.getLogger(__name__)


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


def build_rollout_request(
    *,
    batch: Dict[str, Any],
    samples_per_prompt: int,
    sampling_defaults: Optional[Dict[str, Any]] = None,
) -> RolloutRequest:
    k = int(samples_per_prompt)
    if k < 1:
        raise ValueError(
            f"samples_per_prompt must be >= 1 when building rollout requests, got {k}."
        )
    prompts = batch.get("prompts") or []
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("Rollout request building requires non-empty batch['prompts'].")

    sampling_defaults = dict(sampling_defaults or {})
    base_size = len(prompts)
    expanded_prompts = [prompt for prompt in prompts for _ in range(k)]

    raw_prompt_ids = batch.get("prompt_ids")
    if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == base_size:
        base_prompt_ids = [str(prompt_id) for prompt_id in raw_prompt_ids]
    else:
        base_prompt_ids = [f"prompt:{prompt_idx}" for prompt_idx in range(base_size)]
    expanded_prompt_ids = [prompt_id for prompt_id in base_prompt_ids for _ in range(k)]

    raw_group_ids = batch.get("group_ids")
    resolved_group_ids = _resolve_explicit_group_ids(
        raw_group_ids,
        base_size=base_size,
    )
    if resolved_group_ids is not None:
        base_group_ids = resolved_group_ids
    else:
        base_group_ids = list(base_prompt_ids)
    expanded_group_ids = [group_id for group_id in base_group_ids for _ in range(k)]

    sample_ids = [
        f"prompt:{base_prompt_ids[prompt_idx]}:sample:{replica_idx}"
        for prompt_idx in range(base_size)
        for replica_idx in range(k)
    ]

    init_same_noise = bool(
        batch.get(
            "init_same_noise",
            sampling_defaults.get("init_same_noise", False),
        )
    )

    raw_noise_group_ids = batch.get("noise_group_ids")
    if isinstance(raw_noise_group_ids, list) and len(raw_noise_group_ids) == base_size:
        base_noise_group_ids = [str(group_id) for group_id in raw_noise_group_ids]
    else:
        base_noise_group_ids = list(base_prompt_ids)
    if init_same_noise:
        expanded_noise_group_ids = [gid for gid in base_noise_group_ids for _ in range(k)]
    else:
        expanded_noise_group_ids = list(sample_ids)

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
        sampling_defaults.get("num_inference_steps"),
    )
    raw_guidance_scale = batch.get(
        "guidance_scale",
        sampling_defaults.get("guidance_scale"),
    )
    raw_height = batch.get("height", sampling_defaults.get("height"))
    raw_width = batch.get("width", sampling_defaults.get("width"))
    raw_num_frames = batch.get("num_frames", sampling_defaults.get("num_frames"))
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
        sampling_defaults.get("sampling_adapter"),
    )
    raw_seed = batch.get("seed", sampling_defaults.get("seed"))
    seed = None if raw_seed is None else int(raw_seed)
    request_kwargs = batch.get("kwargs")
    if isinstance(request_kwargs, dict):
        request_kwargs = dict(request_kwargs)
    else:
        request_kwargs = {}

    return RolloutRequest(
        prompts=expanded_prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        sampling={
            "seed": seed,
            "init_same_noise": init_same_noise,
            "samples_per_prompt": k,
            "sampling_adapter": sampling_adapter,
            "kwargs": request_kwargs,
        },
        meta={
            "prompt_ids": expanded_prompt_ids,
            "sample_ids": sample_ids,
            "group_ids": expanded_group_ids,
            "noise_group_ids": expanded_noise_group_ids,
            "prompt_metadata": expanded_prompt_metadata,
        },
        inputs={"latents": latents} if latents is not None else {},
    )


def estimate_request_batches(
    *,
    prompt_count: int,
    samples_per_prompt: int,
    is_direct_sampling_mode: bool,
    max_samples_per_request: int = 0,
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
        not is_direct_sampling_mode
        or max_samples_per_request == 0
        or max_samples_per_request >= total_samples
    ):
        return 1
    return (total_samples + max_samples_per_request - 1) // max_samples_per_request


def plan_request_batches(
    *,
    batch: Dict[str, Any],
    samples_per_prompt: int,
    is_direct_sampling_mode: bool,
    max_samples_per_request: Optional[int],
    sampling_defaults: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, RolloutRequest]]:
    prompts = batch.get("prompts", []) or []
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("Rollout request building requires non-empty batch['prompts'].")

    requests_per_rollout = estimate_request_batches(
        prompt_count=len(prompts),
        samples_per_prompt=samples_per_prompt,
        is_direct_sampling_mode=is_direct_sampling_mode,
        max_samples_per_request=max_samples_per_request,
    )
    if requests_per_rollout <= 1:
        return [
            (
                0,
                build_rollout_request(
                    batch=batch,
                    samples_per_prompt=samples_per_prompt,
                    sampling_defaults=sampling_defaults,
                ),
            )
        ]

    full_request = build_rollout_request(
        batch=batch,
        samples_per_prompt=samples_per_prompt,
        sampling_defaults=sampling_defaults,
    )
    request_batch_size = len(full_request.prompts)
    if max_samples_per_request is None or int(max_samples_per_request) < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")
    max_per_request = int(max_samples_per_request)
    requests: List[Tuple[int, RolloutRequest]] = []
    for batch_index, start in enumerate(range(0, request_batch_size, max_per_request)):
        end = min(start + max_per_request, request_batch_size)
        requests.append(
            (
                batch_index,
                full_request.slice_prompts(start, end),
            )
        )
    return requests


@dataclass(frozen=True)
class InflightRequestBatch:
    """One launched sampling request not yet resolved."""

    batch_index: int
    request: RolloutRequest
    future: Any


@dataclass(frozen=True)
class InflightRolloutRequests:
    """All launched sampling requests for one rollout."""

    inflight_requests: List[InflightRequestBatch]
    rollout_id: Optional[int] = None


def _aggregate_request_results(
    *,
    request_results: List[Tuple[RolloutRequest, List[Any]]],
) -> Tuple[RolloutRequest, List[Any]]:
    """Aggregate per-request sampled results into one request + output list."""
    all_outputs: List[Any] = []
    requests: List[RolloutRequest] = []

    for request, sampler_outputs in request_results:
        all_outputs.extend(sampler_outputs)
        requests.append(request)

    return RolloutRequest.concat(requests), all_outputs


def execute_request_batches(
    *,
    request_batches: Optional[List[Tuple[int, RolloutRequest]]],
    execute_sampling_request: Callable[[RolloutRequest], List[Any]],
    validate_sampler_outputs: Optional[Callable[[List[Any]], None]] = None,
    rollout_id: Optional[int] = None,
) -> Tuple[RolloutRequest, List[Any]]:
    """Execute one rollout, optionally split into multiple typed requests."""
    planned_requests = list(request_batches or [])
    if not planned_requests:
        raise ValueError("Rollout request planner produced no sampling requests.")

    requests_per_rollout = len(planned_requests)
    rank = int(os.environ.get("RANK", 0))
    rollout_seed_offset = 0 if rollout_id is None else int(rollout_id)
    request_results: List[Tuple[RolloutRequest, List[Any]]] = []

    for batch_idx, current_request in tqdm(
        planned_requests,
        desc=f"Rollout {rollout_id} on rank {rank}",
        unit="request",
        disable=(rank != 0),
    ):
        if not isinstance(current_request.prompts, list) or len(current_request.prompts) == 0:
            raise ValueError("Rollout request executor requires non-empty request.prompts.")
        current_request = current_request.with_seed_offset(rollout_seed_offset)

        if requests_per_rollout > 1:
            logger.debug(
                "rollout=%s direct-sampling request %s/%s samples=%s",
                rollout_id,
                batch_idx + 1,
                requests_per_rollout,
                len(current_request.prompts),
            )

        _req_t0 = _time.perf_counter()
        sampler_outputs = execute_sampling_request(request=current_request)
        _req_t1 = _time.perf_counter()

        if validate_sampler_outputs is not None:
            validate_sampler_outputs(sampler_outputs=sampler_outputs)
        _req_t2 = _time.perf_counter()
        logger.debug(
            "[TIMING] sample_rollout req=%d/%d: sample=%.2fs validate=%.2fs",
            batch_idx + 1,
            requests_per_rollout,
            _req_t1 - _req_t0,
            _req_t2 - _req_t1,
        )
        request_results.append((current_request, sampler_outputs))

    return _aggregate_request_results(
        request_results=request_results,
    )


def launch_request_batches_async(
    *,
    request_batches: Optional[List[Tuple[int, RolloutRequest]]],
    launch_sampling_request: Callable[[RolloutRequest], InflightRequestBatch | Tuple[RolloutRequest, Any]],
    rollout_id: Optional[int] = None,
) -> InflightRolloutRequests:
    """Launch one rollout's sampling requests without waiting for completion."""
    planned_requests = list(request_batches or [])
    if not planned_requests:
        raise ValueError("Rollout request planner produced no sampling requests.")

    inflight_requests: List[InflightRequestBatch] = []
    requests_per_rollout = len(planned_requests)
    rank = int(os.environ.get("RANK", 0))
    rollout_seed_offset = 0 if rollout_id is None else int(rollout_id)

    for batch_idx, current_request in tqdm(
        planned_requests,
        desc=f"Launch rollout {rollout_id} on rank {rank}",
        unit="request",
        disable=(rank != 0),
    ):
        if not isinstance(current_request.prompts, list) or len(current_request.prompts) == 0:
            raise ValueError("Rollout request executor requires non-empty request.prompts.")
        current_request = current_request.with_seed_offset(rollout_seed_offset)

        if requests_per_rollout > 1:
            logger.debug(
                "launch rollout=%s direct-sampling request %s/%s samples=%s",
                rollout_id,
                batch_idx + 1,
                requests_per_rollout,
                len(current_request.prompts),
            )

        launched = launch_sampling_request(request=current_request)
        if isinstance(launched, InflightRequestBatch):
            inflight_requests.append(launched)
        else:
            launched_request, future = launched
            inflight_requests.append(
                InflightRequestBatch(
                    batch_index=batch_idx,
                    request=launched_request,
                    future=future,
                )
            )

    return InflightRolloutRequests(
        inflight_requests=inflight_requests,
        rollout_id=rollout_id,
    )


def resolve_request_batches_async(
    *,
    inflight_rollout: InflightRolloutRequests,
    resolve_sampling_request: Callable[[Any], List[Any]],
    validate_sampler_outputs: Optional[Callable[[List[Any]], None]] = None,
) -> Tuple[RolloutRequest, List[Any]]:
    """Resolve previously launched sampling requests and aggregate outputs."""
    request_results: List[Tuple[RolloutRequest, List[Any]]] = []
    requests_per_rollout = len(inflight_rollout.inflight_requests)

    for inflight in inflight_rollout.inflight_requests:
        _req_t0 = _time.perf_counter()
        sampler_outputs = resolve_sampling_request(inflight.future)
        _req_t1 = _time.perf_counter()

        if validate_sampler_outputs is not None:
            validate_sampler_outputs(sampler_outputs=sampler_outputs)
        _req_t2 = _time.perf_counter()
        logger.debug(
            "[TIMING] resolve_rollout req=%d/%d: ray_get=%.2fs validate=%.2fs",
            inflight.batch_index + 1,
            requests_per_rollout,
            _req_t1 - _req_t0,
            _req_t2 - _req_t1,
        )
        request_results.append((inflight.request, sampler_outputs))

    return _aggregate_request_results(
        request_results=request_results,
    )


def distributed_sample(
    *,
    actor_group: Any,
    request: RolloutRequest,
) -> List[RolloutSamples]:
    """Sample across distributed rollout actors."""
    if actor_group is None:
        raise RuntimeError("No sampling actors available")

    prompts = request.prompts
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    outputs = actor_group.generate(request)
    return normalize_rollout_outputs(outputs)


def normalize_rollout_outputs(outputs: List[Any]) -> List[RolloutSamples]:
    """Normalize raw actor-group generate outputs into flat RolloutSamples rows."""
    merged_outputs: List[RolloutSamples] = []
    for output in outputs:
        if isinstance(output, RolloutSamples):
            merged_outputs.append(output)
            continue

        if isinstance(output, (list, tuple)):
            for item in output:
                if not isinstance(item, RolloutSamples):
                    raise TypeError(
                        "Sampling stage expects RolloutSamples from actors, "
                        f"got {type(item).__name__} inside {type(output).__name__}."
                    )
                merged_outputs.append(item)
            continue

        raise TypeError(
            "Sampling stage expects RolloutSamples from actors, "
            f"got {type(output).__name__}."
        )

    return merged_outputs


def validate_sampler_outputs_against_contract(
    *,
    sampler_outputs: List[Any],
    requirements: Any,
    allow_replay: bool,
    assert_step_alignment: bool,
    mode_label: str,
) -> None:
    """Validate rollout outputs against the algorithm's declared sampling contract."""
    replay_notice_emitted = False
    for idx, out in enumerate(sampler_outputs):
        try:
            meta = out.aux.get("metadata") if hasattr(out, "aux") else {}
            meta = meta or {}
            generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
            allow_missing_log_probs = bool(allow_replay)
            if allow_missing_log_probs and not replay_notice_emitted:
                logger.warning(
                    "Replay path enabled: allowing missing rollout log_probs; "
                    "training actors will replay old log_probs before backward."
                )
                replay_notice_emitted = True

            out.validate_contract(
                requires_log_probs=bool(requirements.requires_log_prob) and not allow_missing_log_probs,
                requires_trajectory=bool(requirements.requires_trajectory),
                requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
            )

            if assert_step_alignment:
                resolved_steps = out.aux.get("step_indices")
                if resolved_steps is None:
                    resolved_steps = torch.arange(
                        out.timesteps.shape[0], device=out.timesteps.device, dtype=torch.long
                    )
                if int(resolved_steps.shape[0]) != int(out.timesteps.shape[0]):
                    raise ValueError(
                        f"step/timestep length mismatch: step_indices={resolved_steps.shape[0]}, "
                        f"timesteps={out.timesteps.shape[0]}"
                    )
        except Exception as exc:
            meta = out.aux.get("metadata") if hasattr(out, "aux") else {}
            meta = meta or {}
            generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
            capabilities = meta.get("engine_capabilities") if isinstance(meta, dict) else None
            trajectories = out.aux.get("trajectories") if hasattr(out, "aux") else None
            step_indices = out.aux.get("step_indices") if hasattr(out, "aux") else None
            traj_shape = tuple(trajectories.shape) if trajectories is not None else None
            latents_shape = tuple(out.latents.shape) if getattr(out, "latents", None) is not None else None
            if step_indices is None and hasattr(out, "timesteps"):
                step_indices = torch.arange(
                    out.timesteps.shape[0], device=out.timesteps.device, dtype=torch.long
                )
            steps_shape = tuple(step_indices.shape) if step_indices is not None else None
            hint = ""
            if generator_type == "sglang":
                hint = (
                    f" {generator_type} currently may omit rollout log_probs; "
                    "set sampling.logprob_source='replay' and ensure prompt text inputs are present."
                )
            raise RuntimeError(
                f"Sampler output contract validation failed in {mode_label} path at index={idx}: {exc}.{hint} "
                f"capabilities={capabilities}, latents_shape={latents_shape}, "
                f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
            ) from exc


def build_sampler_output_validator(
    *,
    requirements: Any,
    validation_config: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Any]], None]:
    """Build a reusable sampler-output validator from algorithm validation config."""
    resolved_config = dict(validation_config or {})
    return partial(
        validate_sampler_outputs_against_contract,
        requirements=requirements,
        allow_replay=bool(resolved_config.get("allow_replay", False)),
        assert_step_alignment=bool(resolved_config.get("assert_step_alignment", True)),
        mode_label=str(resolved_config.get("mode_label", "trajectory")),
    )


def compute_advantages(
    *,
    algorithm: Any,
    rewards: torch.Tensor,
    group_ids: Optional[List[str]] = None,
    component_rewards: Optional[Dict[str, List[float]]] = None,
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Delegate reward-component-aware advantage semantics to the algorithm."""
    if component_rewards is not None and reward_components is not None:
        raise ValueError(
            "compute_advantages accepts either component_rewards or reward_components, not both."
        )
    resolved_component_rewards = (
        component_rewards if component_rewards is not None else reward_components
    )
    return algorithm.compute_advantages_with_components(
        rewards=rewards,
        group_ids=group_ids,
        reward_components=resolved_component_rewards,
        reward_component_weights=reward_component_weights,
    )


__all__ = [
    "InflightRequestBatch",
    "InflightRolloutRequests",
    "build_sampler_output_validator",
    "build_rollout_request",
    "compute_advantages",
    "distributed_sample",
    "estimate_request_batches",
    "execute_request_batches",
    "launch_request_batches_async",
    "normalize_rollout_outputs",
    "plan_request_batches",
    "resolve_request_batches_async",
    "validate_sampler_outputs_against_contract",
]
