"""
Rollout pipeline — unified stage helpers for RolloutManager.

This module merges all five pipeline stages into a single file for reduced
file-jumping when reading the rollout→train data flow:

- **Sampling stage**: expand_batch_for_sampling, distributed_sample
- **Reward stage**: extract_images_from_output, extract_videos_from_output,
  reward_prefers_video_inputs, compute_rewards
- **Advantage stage**: get_reward_component_weights, compute_advantages
- **Assemble stage**: assemble_forward_training_batch, assemble_backward_training_batch
- **Partition stage**: maybe_partition_training_batch
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sampling import LogProbData, PromptEmbeddings, RolloutOutput, RolloutRequest
from diffusionrl.types.training_batch import BackwardTrainingBatch, ForwardTrainingBatch

logger = logging.getLogger(__name__)


# =========================================================================
# Sampling stage
# =========================================================================

def expand_batch_for_sampling(
    batch: Dict[str, Any],
    *,
    num_samples_per_prompt: int,
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """
    Expand batch for K-repeat sampling using prompt-major order.

    This repeats prompts along the batch dimension so that
    sampling generates num_samples_per_prompt outputs per unique prompt.

    Returns:
        (expanded_batch, train_prompts)
    """
    k = int(num_samples_per_prompt)
    if k <= 1:
        return batch, batch.get("prompts")

    prompts = batch.get("prompts")
    base_size = len(prompts) if prompts is not None else None

    if base_size is None or base_size == 0:
        return batch, prompts

    train_prompts: Optional[List[str]] = None
    if prompts is not None:
        train_prompts = [p for p in prompts for _ in range(k)]

    expanded: Dict[str, Any] = dict(batch)
    if prompts is not None:
        expanded["prompts"] = train_prompts
    if "metadata" in expanded and isinstance(expanded["metadata"], list):
        metadata = expanded["metadata"]
        if len(metadata) == base_size:
            expanded["metadata"] = [m for m in metadata for _ in range(k)]

    def _repeat(value: Any) -> Any:
        if torch.is_tensor(value) and value.shape[0] == base_size:
            return value.repeat_interleave(k, dim=0)
        return value

    for key in ("latents",):
        if key in expanded:
            expanded[key] = _repeat(expanded[key])

    return expanded, train_prompts


def distributed_sample(
    *,
    actor_group: Any,
    batch: Dict[str, Any],
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    init_same_noise: bool,
    num_samples_per_prompt: int,
    sde_indices: Optional[Set[int]] = None,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
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

    prompts = batch.get("prompts", [])
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    extra_kwargs: Dict[str, Any] = {}
    extra_kwargs["init_same_noise"] = init_same_noise
    extra_kwargs["num_samples_per_prompt"] = num_samples_per_prompt
    if isinstance(extra_generate_kwargs, dict) and extra_generate_kwargs:
        extra_kwargs.update(extra_generate_kwargs)

    request = RolloutRequest(
        prompts=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        sde_indices=sde_indices,
        decode_for_reward=True,
        latents=batch.get("latents"),
        kwargs=extra_kwargs,
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


def compute_rewards(
    *,
    reward_service: Any,
    reward_path: str,
    num_samples_per_prompt: int,
    sampler_outputs: List[RolloutOutput],
    prompts: List[str],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Compute rewards for generated samples using batch processing.
    Uses RewardService to compute rewards in a batched manner.

    Args:
        sampler_outputs: List of sampler outputs
        prompts: List of text prompts

    Returns:
        Tuple of:
            - Tensor of rewards [batch_size]
            - Reward components by worker/model name
    """
    all_images: List[Any] = []
    all_videos: List[torch.Tensor] = []
    all_prompts: List[str] = []
    all_metadata: List[Optional[Dict[str, Any]]] = []

    sample_idx = 0

    def _append_media(items: List[Any], target: List[Any]) -> None:
        nonlocal sample_idx
        for item in items:
            prompt_idx = sample_idx // num_samples_per_prompt
            prompt = prompts[prompt_idx % len(prompts)] if prompts else ""
            target.append(item)
            all_prompts.append(prompt)
            if prompt_metadata and len(prompt_metadata) > 0:
                all_metadata.append(prompt_metadata[prompt_idx % len(prompt_metadata)])
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
) -> Dict[str, float]:
    """Map reward component name to configured worker weight."""
    default_weights = {name: 1.0 for name in reward_components.keys()}
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
    num_samples_per_prompt: int,
    reward_mix_mode: str,
    rewards: torch.Tensor,
    prompts: List[str],
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_workers: Optional[Iterable[Any]] = None,
) -> torch.Tensor:
    """
    Compute advantages from reward tensor.

    Default path (`reward_mix_mode=reward_aggr`) uses aggregated rewards directly.
    Optional path (`reward_mix_mode=advantage_aggr`) computes advantages per reward
    component and aggregates them with reward worker weights.
    """
    if reward_mix_mode != "advantage_aggr" or not reward_components:
        return algorithm.compute_advantages(
            rewards=rewards,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )

    weights = get_reward_component_weights(reward_components, reward_workers)
    weighted_advantages = torch.zeros_like(rewards)
    total_weight = 0.0

    for component_name, component_rewards in reward_components.items():
        component_tensor = torch.tensor(
            component_rewards,
            dtype=rewards.dtype,
            device=rewards.device,
        )
        if component_tensor.shape != rewards.shape:
            logger.warning(
                "Skipping reward component %s due to shape mismatch: expected %s, got %s",
                component_name,
                tuple(rewards.shape),
                tuple(component_tensor.shape),
            )
            continue

        component_advantages = algorithm.compute_advantages(
            rewards=component_tensor,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )
        weight = float(weights.get(component_name, 1.0))
        weighted_advantages += component_advantages * weight
        total_weight += weight

    if total_weight <= 0:
        logger.warning(
            "reward_mix_mode=advantage_aggr but no valid reward components; "
            "falling back to aggregated reward advantages."
        )
        return algorithm.compute_advantages(
            rewards=rewards,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )

    return weighted_advantages / total_weight


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
    """Convert pipeline outputs to NFT training data format."""
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
    """Convert pipeline outputs to trajectory-based training batch."""
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
        trajectories_tensor = torch.cat(trajectories, dim=0)
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
