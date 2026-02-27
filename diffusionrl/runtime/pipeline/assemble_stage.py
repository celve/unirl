"""Training batch assembly stage helpers for RolloutManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import torch

from diffusionrl.types.sampling import LogProbData, PromptEmbeddings, RolloutOutput
from diffusionrl.types.training_batch import BackwardTrainingBatch, ForwardTrainingBatch

logger = logging.getLogger(__name__)


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
