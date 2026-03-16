"""
Rollout pipeline — unified stage helpers for RolloutManager.

This module merges all five pipeline stages into a single file for reduced
file-jumping when reading the rollout→train data flow:

- **Sampling stage**: distributed_sample
- **Reward stage**: extract_images_from_output, extract_videos_from_output,
  reward_prefers_video_inputs, compute_rewards
- **Advantage stage**: get_reward_component_weights, compute_advantages
- **Assemble stage**: assemble_forward_training_batch, assemble_backward_training_batch
- **Partition stage**: maybe_partition_training_batch
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sampling import LogProbData, PromptEmbeddings, RolloutOutput, RolloutRequest
from diffusionrl.types.training_batch import BackwardTrainingBatch, ForwardTrainingBatch

logger = logging.getLogger(__name__)


# =========================================================================
# Sampling stage
# =========================================================================


def distributed_sample(
    *,
    actor_group: Any,
    request: RolloutRequest,
) -> List[RolloutOutput]:
    """
    Sample across distributed rollout actors.

    This is the natural construction point where scattered parameters are
    bundled into a :class:`RolloutRequest` before being dispatched.

    Args:
        batch: Batch containing text prompts (prompt-only input contract)
        sde_indices: Set of timestep indices for SDE sampling (MixGRPO).
            If None, all timesteps use SDE (standard GRPO).

    Returns:
        List of RolloutOutput.
    """
    if actor_group is None:
        raise RuntimeError("No sampling actors available")

    prompts = request.prompts
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    outputs = actor_group.generate(request)

    merged_outputs: List[RolloutOutput] = []
    for output in outputs:
        if isinstance(output, RolloutOutput):
            merged_outputs.append(output)
            continue

        if isinstance(output, (list, tuple)):
            for item in output:
                if not isinstance(item, RolloutOutput):
                    raise TypeError(
                        "Sampling stage expects RolloutOutput from actors, "
                        f"got {type(item).__name__} inside {type(output).__name__}."
                    )
                merged_outputs.append(item)
            continue

        raise TypeError(
            "Sampling stage expects RolloutOutput from actors, "
            f"got {type(output).__name__}."
        )

    return merged_outputs


# =========================================================================
# Reward stage
# =========================================================================

def extract_images_from_output(output: RolloutOutput) -> List[Any]:
    """
    Extract images from a sampler output.

    Args:
        output: Typed sampler output.

    Returns:
        List of images (PIL.Image or tensors)
    """
    if not isinstance(output, RolloutOutput):
        raise TypeError(
            "Reward stage expects RolloutOutput, "
            f"got {type(output).__name__}."
        )

    if output.decoded_images is not None:
        return output.decoded_images if isinstance(output.decoded_images, list) else [output.decoded_images]

    if isinstance(output.latents, torch.Tensor) and output.latents.dim() >= 3:
        return [lat for lat in output.latents]
    if output.latents is not None:
        return [output.latents]

    return []


def extract_videos_from_output(output: RolloutOutput) -> List[torch.Tensor]:
    """Extract decoded videos (preferred) or 5D tensors as video payload."""
    if not isinstance(output, RolloutOutput):
        raise TypeError(
            "Reward stage expects RolloutOutput, "
            f"got {type(output).__name__}."
        )

    decoded_videos = None
    metadata = output.metadata
    if isinstance(metadata, dict):
        decoded_videos = metadata.get("decoded_videos")
    latents = output.latents

    if torch.is_tensor(decoded_videos):
        if decoded_videos.dim() >= 5:
            return [video for video in decoded_videos]
        if decoded_videos.dim() == 4:
            return [decoded_videos]

    if torch.is_tensor(latents):
        if latents.dim() >= 5:
            return [video for video in latents]
        if latents.dim() == 4:
            return [latents]

    return []


def reward_prefers_video_inputs(reward_path: str) -> bool:
    """Best-effort switch for video-native reward workers."""
    reward_path = str(reward_path or "")
    return "VideoRewardWorker" in reward_path


def _read_precomputed_reward_payload(output: RolloutOutput) -> Optional[Tuple[List[float], Dict[str, List[float]]]]:
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    raw_rewards = metadata.get("precomputed_rewards")
    if raw_rewards is None:
        return None
    rewards = [float(v) for v in list(raw_rewards)]
    reward_components = dict(metadata.get("precomputed_reward_components") or {})
    normalized_components: Dict[str, List[float]] = {}
    for name, values in reward_components.items():
        normalized_components[str(name)] = [float(v) for v in list(values or [])]
    return rewards, normalized_components


def collect_precomputed_rewards(
    *,
    sampler_outputs: List[RolloutOutput],
) -> Optional[Tuple[torch.Tensor, Dict[str, List[float]]]]:
    """Collect precomputed rewards from rollout outputs when available."""
    if not sampler_outputs:
        return None

    saw_precomputed = False
    saw_missing = False
    all_rewards: List[float] = []
    reward_components: Dict[str, List[float]] = {}

    for output in sampler_outputs:
        payload = _read_precomputed_reward_payload(output)
        if payload is None:
            saw_missing = True
            continue
        saw_precomputed = True
        rewards, components = payload
        if len(rewards) != int(output.batch_size):
            raise ValueError(
                "Precomputed rewards length must match rollout output batch_size. "
                f"Got rewards={len(rewards)} batch_size={int(output.batch_size)}."
            )
        all_rewards.extend(rewards)
        for name, values in components.items():
            if len(values) != len(rewards):
                raise ValueError(
                    "Precomputed reward component length must match reward length. "
                    f"Got component={name} len={len(values)} rewards={len(rewards)}."
                )
            reward_components.setdefault(name, []).extend(values)

    if saw_precomputed and saw_missing:
        raise ValueError(
            "Mixed rollout reward execution is not supported: some sampler outputs contain "
            "precomputed rewards while others do not."
        )
    if not saw_precomputed:
        return None
    return torch.tensor(all_rewards, dtype=torch.float32), reward_components


def normalize_prompt_metadata(
    *,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    samples_per_prompt: Optional[int] = None,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Normalize prompt metadata to sample-aligned layout.

    Preferred path:
    - prompt metadata is already sample-aligned
    - or prompt metadata is prompt-aligned and prompt_ids are provided so
      metadata can be expanded by explicit prompt identity

    Legacy fallback:
    - expand by fixed ``samples_per_prompt`` when explicit ids are absent
    """
    if not isinstance(prompt_metadata, list) or not prompt_metadata:
        return None

    sample_count = len(prompts)
    if sample_count <= 0:
        return None

    if len(prompt_metadata) == sample_count:
        return list(prompt_metadata)

    if isinstance(prompt_ids, list) and len(prompt_ids) == sample_count:
        ordered_prompt_ids: List[str] = []
        seen: set[str] = set()
        for raw_prompt_id in prompt_ids:
            prompt_id = str(raw_prompt_id).strip()
            if not prompt_id or prompt_id in seen:
                continue
            seen.add(prompt_id)
            ordered_prompt_ids.append(prompt_id)
        if len(prompt_metadata) == len(ordered_prompt_ids):
            metadata_by_prompt_id = {
                prompt_id: prompt_metadata[idx]
                for idx, prompt_id in enumerate(ordered_prompt_ids)
            }
            return [metadata_by_prompt_id.get(str(raw_prompt_id).strip()) for raw_prompt_id in prompt_ids]

    k = max(1, int(samples_per_prompt or 1))
    if len(prompt_metadata) * k == sample_count:
        return [
            metadata
            for metadata in prompt_metadata
            for _ in range(k)
        ]

    logger.warning(
        "Ignoring misaligned prompt metadata: prompts=%s metadata=%s prompt_ids=%s samples_per_prompt=%s",
        sample_count,
        len(prompt_metadata),
        len(prompt_ids) if isinstance(prompt_ids, list) else None,
        samples_per_prompt,
    )
    return None


def compute_rewards(
    *,
    reward_service: Any,
    reward_path: str,
    samples_per_prompt: int,
    sampler_outputs: List[RolloutOutput],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    group_ids: Optional[List[str]] = None,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Compute rewards for generated samples using batch processing.
    Uses RewardService to compute rewards in a batched manner.

    Args:
        sampler_outputs: List of sampler outputs
        prompts: Sample-aligned prompt list. The reward stage uses a single
            prompt representation everywhere; prompt-major base prompts are
            expanded by the caller before entering this function.

    Returns:
        Tuple of:
            - Tensor of rewards [batch_size]
            - Reward components by worker/model name
    """
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "compute_rewards requires a non-empty sample-aligned prompts list."
        )

    _rw_t0 = _time.perf_counter()
    precomputed = collect_precomputed_rewards(sampler_outputs=sampler_outputs)
    if precomputed is not None:
        _rw_t1 = _time.perf_counter()
        logger.warning(
            "[TIMING] compute_rewards: precomputed_collect=%.2fs",
            _rw_t1 - _rw_t0,
        )
        return precomputed
    if reward_service is None:
        raise RuntimeError(
            "RewardService is not initialized and sampler outputs do not include precomputed rewards."
        )

    all_images: List[Any] = []
    all_videos: List[torch.Tensor] = []
    all_prompts: List[str] = []
    all_prompt_ids: List[str] = []
    all_sample_ids: List[str] = []
    all_group_ids: List[str] = []
    all_metadata: List[Optional[Dict[str, Any]]] = []

    sample_idx = 0

    normalized_prompt_metadata = normalize_prompt_metadata(
        prompt_metadata=prompt_metadata,
        prompts=prompts,
        prompt_ids=prompt_ids,
        samples_per_prompt=samples_per_prompt,
    )

    def _append_media(items: List[Any], target: List[Any]) -> None:
        nonlocal sample_idx
        for item in items:
            if sample_idx >= len(prompts):
                raise IndexError(
                    "Reward media count exceeded prompt count while assembling RewardRequest. "
                    f"sample_idx={sample_idx}, prompts={len(prompts)}"
                )
            prompt = prompts[sample_idx]
            target.append(item)
            all_prompts.append(prompt)
            if prompt_ids is not None and sample_idx < len(prompt_ids):
                all_prompt_ids.append(str(prompt_ids[sample_idx]))
            if sample_ids is not None and sample_idx < len(sample_ids):
                all_sample_ids.append(str(sample_ids[sample_idx]))
            if group_ids is not None and sample_idx < len(group_ids):
                all_group_ids.append(str(group_ids[sample_idx]))
            if normalized_prompt_metadata is not None:
                all_metadata.append(normalized_prompt_metadata[sample_idx])
            else:
                all_metadata.append(None)
            sample_idx += 1

    prefer_video_inputs = reward_prefers_video_inputs(reward_path)
    if prefer_video_inputs:
        all_videos = []
        for output in sampler_outputs:
            videos = extract_videos_from_output(output)
            if not videos:
                prefer_video_inputs = False
                all_videos = []
                all_prompts = []
                all_metadata = []
                sample_idx = 0
                break
            _append_media(videos, all_videos)

    if not prefer_video_inputs:
        for output in sampler_outputs:
            images = extract_images_from_output(output)
            _append_media(images, all_images)

    if not all_images and not all_videos:
        logger.warning("No media extracted from sampler outputs")
        sample_count = 0
        for output in sampler_outputs:
            if isinstance(output, RolloutOutput):
                sample_count += int(output.batch_size)
        if sample_count <= 0:
            sample_count = len(prompts)
        return torch.zeros(sample_count, dtype=torch.float32), {}

    request_kwargs: Dict[str, Any] = {
        "prompts": all_prompts,
        "metadata": all_metadata if any(m is not None for m in all_metadata) else None,
    }
    if len(all_prompt_ids) == len(all_prompts):
        request_kwargs["prompt_ids"] = all_prompt_ids
    if len(all_sample_ids) == len(all_prompts):
        request_kwargs["sample_ids"] = all_sample_ids
    if len(all_group_ids) == len(all_prompts):
        request_kwargs["group_ids"] = all_group_ids
    if prefer_video_inputs and all_videos:
        request_kwargs["videos"] = all_videos
    else:
        request_kwargs["images"] = all_images
    request = RewardRequest(**request_kwargs)

    response = reward_service.compute_rewards(request)
    return torch.tensor(response.rewards, dtype=torch.float32), response.reward_components


# =========================================================================
# Advantage stage
# =========================================================================

def get_reward_component_weights(
    reward_components: Dict[str, List[float]],
    reward_workers: Optional[Iterable[Any]] = None,
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Map reward component name to configured worker weight."""
    default_weights = {name: 1.0 for name in reward_components.keys()}
    if reward_component_weights:
        for name, value in reward_component_weights.items():
            if name in default_weights:
                default_weights[name] = float(value)
    if reward_workers is None:
        return default_weights

    for worker in reward_workers:
        model_name = worker.get_model_name()
        if model_name in default_weights:
            default_weights[model_name] = float(worker.get_weight())
    return default_weights


def compute_advantages(
    *,
    algorithm: Any,
    component_mix_stage: str,
    rewards: torch.Tensor,
    group_ids: Optional[List[str]] = None,
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_workers: Optional[Iterable[Any]] = None,
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute advantages from reward tensor.

    Delegates reward-mixing and advantage policy to the algorithm.
    """
    resolved_weights = reward_component_weights
    if reward_workers is not None and reward_components:
        resolved_weights = get_reward_component_weights(
            reward_components, reward_workers, reward_component_weights
        )
    return algorithm.compute_advantages_with_components(
        rewards=rewards,
        group_ids=group_ids,
        component_mix_stage=component_mix_stage,
        reward_components=reward_components,
        reward_component_weights=resolved_weights,
    )


# =========================================================================
# Assemble stage
# =========================================================================

def assemble_forward_training_batch(
    *,
    sampler_outputs: List[RolloutOutput],
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    prompts: List[str],
) -> ForwardTrainingBatch:
    """Convert pipeline outputs to NFT training data format.

    .. deprecated::
        Assembly logic has been moved into ``BaseAlgorithm._assemble_forward_batch``.
        This function is kept for backward compatibility but will be removed in a
        future version. Prefer ``algorithm.assemble_training_batch()`` instead.
    """
    clean_latents = []
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_encoder_attention_mask = []
    all_negative_prompt_embeds = []
    all_negative_pooled_prompt_embeds = []
    all_text_ids = []
    all_image_ids = []
    timesteps: Optional[torch.Tensor] = None

    for idx, output in enumerate(sampler_outputs):
        if not isinstance(output, RolloutOutput):
            raise TypeError(
                "Assemble stage expects RolloutOutput, "
                f"got {type(output).__name__} at index={idx}."
            )
        if output.embeddings is None:
            raise ValueError(f"RolloutOutput at index={idx} missing embeddings in NFT path.")

        clean_latents.append(output.latents)

        emb = output.embeddings
        all_prompt_embeds.append(emb.prompt_embeds)
        if emb.pooled_prompt_embeds is not None:
            all_pooled_prompt_embeds.append(emb.pooled_prompt_embeds)
        if emb.encoder_attention_mask is not None:
            all_encoder_attention_mask.append(emb.encoder_attention_mask)
        if emb.negative_prompt_embeds is not None:
            all_negative_prompt_embeds.append(emb.negative_prompt_embeds)
        if emb.negative_pooled_prompt_embeds is not None:
            all_negative_pooled_prompt_embeds.append(emb.negative_pooled_prompt_embeds)
        if emb.text_ids is not None:
            all_text_ids.append(emb.text_ids)
        if emb.image_ids is not None:
            all_image_ids.append(emb.image_ids)

        if timesteps is None:
            timesteps = output.timesteps
        elif not torch.equal(timesteps.to(output.timesteps.device), output.timesteps):
            raise ValueError("Mismatched timesteps across sampler outputs")

    if clean_latents:
        clean_latents_tensor = torch.cat(clean_latents, dim=0)
    else:
        raise ValueError("No clean latents found in sampler outputs")

    prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
    pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None
    encoder_attention_mask = torch.cat(all_encoder_attention_mask, dim=0) if all_encoder_attention_mask else None
    negative_prompt_embeds = torch.cat(all_negative_prompt_embeds, dim=0) if all_negative_prompt_embeds else None
    negative_pooled_prompt_embeds = (
        torch.cat(all_negative_pooled_prompt_embeds, dim=0) if all_negative_pooled_prompt_embeds else None
    )
    text_ids = torch.cat(all_text_ids, dim=0) if all_text_ids else None
    image_ids = all_image_ids[0] if all_image_ids else None

    if prompt_embeds is None:
        raise ValueError("No prompt embeddings found in sampler outputs")
    if timesteps is None:
        raise ValueError("No timesteps found in sampler outputs")

    embeddings = PromptEmbeddings(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        encoder_attention_mask=encoder_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        text_ids=text_ids,
        image_ids=image_ids,
    )

    batch = ForwardTrainingBatch(
        clean_latents=clean_latents_tensor,
        advantages=advantages,
        embeddings=embeddings,
        rewards=rewards,
        prompts=prompts,
        timesteps=timesteps,
    )
    batch.validate()
    return batch


def assemble_backward_training_batch(
    *,
    algorithm: Any,
    num_inference_steps: int,
    sampler_outputs: List[RolloutOutput],
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    prompts: List[str],
    sde_indices: Optional[Set[int]] = None,
) -> BackwardTrainingBatch:
    """Convert pipeline outputs to trajectory-based training batch.

    .. deprecated::
        Assembly logic has been moved into ``BaseAlgorithm._assemble_backward_batch``.
        This function is kept for backward compatibility but will be removed in a
        future version. Prefer ``algorithm.assemble_training_batch()`` instead.
    """
    trajectories = []
    log_probs_dicts = []
    timesteps = None
    step_indices = None
    raw_scheduler_indices = (
        {int(i) for i in sde_indices}
        if sde_indices is not None
        else None
    )
    final_sde_indices: Set[int] = set(raw_scheduler_indices or set())
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_encoder_attention_mask = []
    all_negative_prompt_embeds = []
    all_negative_pooled_prompt_embeds = []
    all_text_ids = []
    all_image_ids = []

    for idx, output in enumerate(sampler_outputs):
        if not isinstance(output, RolloutOutput):
            raise TypeError(
                "Assemble stage expects RolloutOutput, "
                f"got {type(output).__name__} at index={idx}."
            )
        if output.trajectories is None:
            raise ValueError(f"RolloutOutput at index={idx} missing trajectories in GRPO path.")
        if output.embeddings is None:
            raise ValueError(f"RolloutOutput at index={idx} missing embeddings in GRPO path.")

        traj = output.trajectories
        log_probs = output.log_probs.to_dict() if output.log_probs is not None else {}
        ts = output.timesteps
        steps = output.step_indices
        sde_idx = output.sde_indices

        trajectories.append(traj)
        log_probs_dicts.append(log_probs)
        if ts is not None and timesteps is None:
            timesteps = ts
        elif ts is not None and timesteps is not None and not torch.equal(timesteps.to(ts.device), ts):
            raise ValueError("Mismatched timesteps across sampler outputs")
        if steps is not None:
            if step_indices is None:
                step_indices = steps
            elif not torch.equal(step_indices.to(steps.device), steps):
                raise ValueError(
                    "Mismatched step_indices across sampler outputs: "
                    f"expected={step_indices.tolist()} got={steps.tolist()}"
                )
        if sde_indices is None:
            final_sde_indices.update(int(i) for i in sde_idx)

        emb = output.embeddings
        all_prompt_embeds.append(emb.prompt_embeds)
        if emb.pooled_prompt_embeds is not None:
            all_pooled_prompt_embeds.append(emb.pooled_prompt_embeds)
        if emb.encoder_attention_mask is not None:
            all_encoder_attention_mask.append(emb.encoder_attention_mask)
        if emb.negative_prompt_embeds is not None:
            all_negative_prompt_embeds.append(emb.negative_prompt_embeds)
        if emb.negative_pooled_prompt_embeds is not None:
            all_negative_pooled_prompt_embeds.append(emb.negative_pooled_prompt_embeds)
        if emb.text_ids is not None:
            all_text_ids.append(emb.text_ids)
        if emb.image_ids is not None:
            all_image_ids.append(emb.image_ids)

    if trajectories:
        _asm_t0 = _time.perf_counter()
        traj_count = len(trajectories)
        traj_shapes = [tuple(t.shape) for t in trajectories[:3]]
        trajectories_tensor = torch.cat(trajectories, dim=0)
        _asm_t1 = _time.perf_counter()
    else:
        raise ValueError("No trajectories found in sampler outputs")

    if timesteps is None:
        raise ValueError("No timesteps found in sampler outputs")
    if step_indices is None:
        step_indices = torch.arange(
            timesteps.shape[0],
            device=timesteps.device,
            dtype=torch.long,
        )

    step_labels = [int(v) for v in step_indices[:-1].tolist()]
    step_label_set = set(step_labels)

    def _normalize_to_step_labels(indices: Set[int], *, source: str) -> Set[int]:
        if not indices:
            return set()
        if indices.issubset(step_label_set):
            return set(indices)
        mapped = {
            step_labels[i]
            for i in indices
            if 0 <= int(i) < len(step_labels)
        }
        if mapped:
            logger.debug(
                "%s indices look positional; mapped to step labels raw=%s mapped=%s",
                source,
                sorted(indices),
                sorted(mapped),
            )
            return mapped
        logger.warning(
            "%s indices do not match sampled step labels and could not be mapped: "
            "raw=%s, available_labels=%s",
            source,
            sorted(indices),
            sorted(step_labels),
        )
        return set()

    if final_sde_indices:
        final_sde_indices = _normalize_to_step_labels(
            set(int(i) for i in final_sde_indices),
            source="Scheduler/Sampler SDE",
        )

    if hasattr(algorithm, "get_training_indices"):
        raw_train_indices = set(
            int(i) for i in algorithm.get_training_indices(len(step_labels))
        )
        train_indices = _normalize_to_step_labels(
            raw_train_indices,
            source=f"{type(algorithm).__name__}.get_training_indices",
        )
        if not train_indices:
            train_indices = step_label_set
        if sde_indices is None:
            final_sde_indices = train_indices
        else:
            final_sde_indices = final_sde_indices & train_indices
        if len(final_sde_indices) == 0:
            logger.warning(
                "Training timestep set is empty after intersecting scheduler and algorithm indices. "
                "Falling back to algorithm-provided indices."
            )
            final_sde_indices = train_indices if len(train_indices) > 0 else set(step_labels)

    num_steps = len(step_labels)
    final_sde_indices = algorithm.get_filtered_training_indices(final_sde_indices, num_steps)

    if len(final_sde_indices) == 0:
        logger.warning(
            "Training timestep set is empty after algorithm filtering; "
            "falling back to all timesteps."
        )
        final_sde_indices = set(step_labels)

    merged_log_probs: Dict[int, torch.Tensor] = {}
    if log_probs_dicts:
        all_indices: Set[int] = set()
        for lpd in log_probs_dicts:
            all_indices.update(lpd.keys())

        for idx in all_indices:
            values = []
            for lpd in log_probs_dicts:
                if idx in lpd:
                    values.append(lpd[idx])
            if values:
                merged_log_probs[idx] = torch.cat(values, dim=0)

    if final_sde_indices:
        merged_log_probs = {
            int(idx): value
            for idx, value in merged_log_probs.items()
            if int(idx) in set(int(i) for i in final_sde_indices)
        }

    prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
    pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None
    encoder_attention_mask = torch.cat(all_encoder_attention_mask, dim=0) if all_encoder_attention_mask else None
    negative_prompt_embeds = torch.cat(all_negative_prompt_embeds, dim=0) if all_negative_prompt_embeds else None
    negative_pooled_prompt_embeds = (
        torch.cat(all_negative_pooled_prompt_embeds, dim=0) if all_negative_pooled_prompt_embeds else None
    )
    text_ids = torch.cat(all_text_ids, dim=0) if all_text_ids else None
    image_ids = all_image_ids[0] if all_image_ids else None

    if prompt_embeds is None:
        raise ValueError("No prompt embeddings found in sampler outputs")

    embeddings = PromptEmbeddings(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        encoder_attention_mask=encoder_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        text_ids=text_ids,
        image_ids=image_ids,
    )

    batch = BackwardTrainingBatch(
        trajectories=trajectories_tensor,
        log_probs=LogProbData.from_dict(merged_log_probs),
        timesteps=timesteps,
        advantages=advantages,
        embeddings=embeddings,
        rewards=rewards,
        prompts=prompts,
        num_steps=num_inference_steps,
        step_indices=step_indices,
        target_sde_indices=set(int(i) for i in final_sde_indices),
    )
    batch.validate()
    _asm_t2 = _time.perf_counter()
    traj_gb = trajectories_tensor.nelement() * trajectories_tensor.element_size() / 1e9
    logger.warning(
        "[TIMING] assemble_backward: cat_traj=%.2fs total=%.2fs n=%d shapes=%s traj_gb=%.2f",
        _asm_t1 - _asm_t0,
        _asm_t2 - _asm_t0,
        traj_count,
        traj_shapes,
        traj_gb,
    )
    return batch


# =========================================================================
# Partition stage
# =========================================================================

def maybe_partition_training_batch(
    *,
    train_data: Any,
    dp_size: Optional[int],
    partition_train_data: bool = True,
) -> Optional[List[Any]]:
    """
    Optionally partition a typed training batch across training ranks.

    Returns:
        List of per-rank typed batch partitions when partition is applied; otherwise None.
    """
    if not partition_train_data or not dp_size:
        return None

    batch_size = getattr(train_data, "batch_size", None)
    if batch_size is None:
        logger.warning("Training batch does not expose batch_size; skipping partition.")
        return None

    per_rank = batch_size // dp_size
    remainder = batch_size % dp_size

    if per_rank == 0:
        logger.warning(
            "Batch size %d too small for dp_size %d; skipping partition.",
            batch_size,
            dp_size,
        )
        return None

    if remainder != 0:
        logger.warning(
            "Batch size %d not divisible by dp_size %d; dropping %d samples for even partition.",
            batch_size,
            dp_size,
            remainder,
        )

    partitions: List[Any] = []
    for rank in range(dp_size):
        start = rank * per_rank
        end = start + per_rank
        part = train_data.slice(start, end)
        partitions.append(part)
    return partitions
