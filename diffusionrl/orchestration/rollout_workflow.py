"""Rollout workflow helpers and business-chain orchestration.

This module is the semantic home for the rollout business chain:

- prompt batch -> request batching
- sample -> reward -> advantage -> assemble
- rollout-facing metadata/debug assembly

Infrastructure-specific details such as actor handles, placement, and remote
lifecycles stay outside this module and are injected through callbacks.

Reading entrypoint: start at ``RolloutWorkflow.build_training_batch()``.
"""

from __future__ import annotations

from functools import partial
import logging
import time as _time
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.orchestration.request_builder import SampledRolloutBatch
from diffusionrl.reward.pipeline import score_from_rollout_outputs as score_reward_stage
from diffusionrl.types.sampling import RolloutOutput, RolloutRequest
from diffusionrl.types.training_batch import TrainingBatch

logger = logging.getLogger(__name__)


class RolloutWorkflow:
    """Business-chain owner for rollout batch production.

    The workflow owns the readable sample -> reward -> advantage -> assemble
    path, while infrastructure-specific seams are injected through callbacks.
    """

    def __init__(
        self,
        *,
        args: Any,
        algorithm: Any,
        reward_scoring_mode: str,
        reward_service: Any,
        request_builder: Any,
        reward_component_weights: Optional[Dict[str, float]],
        load_prompt_batch_fn: Any,
    ) -> None:
        self.args = args
        self.algorithm = algorithm
        self.reward_scoring_mode = str(reward_scoring_mode)
        self.reward_service = reward_service
        self.request_builder = request_builder
        self.reward_component_weights = dict(reward_component_weights or {})
        self.load_prompt_batch_fn = load_prompt_batch_fn

    def build_training_batch(
        self,
        *,
        rollout_id: int,
        sde_indices: Optional[set[int]],
        requirements: Any,
        execute_sampling_request: Any,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> tuple[TrainingBatch, Optional[Dict[str, Any]]]:
        """Produce one typed training batch from the rollout business chain."""
        batch = self._load_prompt_batch()
        prompts = batch.get("prompts", []) or []
        if sde_indices is not None:
            logger.debug(
                "Received explicit rollout SDE indices: %s",
                sorted(int(i) for i in sde_indices)[:5],
            )

        sampling_overrides: Dict[str, Any] = {
            "_keep_reward_media_for_manager": bool(collect_media_preview),
        }
        validation_config = self.algorithm.get_sampler_validation_config(args=self.args)
        if not isinstance(validation_config, dict):
            validation_config = {}

        request_batches = self.request_builder.build_request_batches(
            batch=batch,
            samples_per_prompt=int(self.algorithm.samples_per_prompt),
        )
        request_num_inference_steps = int(request_batches[0][1].num_inference_steps)

        sample_t0 = _time.perf_counter()
        sampled_rollout: SampledRolloutBatch = self.request_builder.execute_request_batches(
            request_batches=request_batches,
            rollout_id=rollout_id,
            execute_sampling_request=partial(
                execute_sampling_request,
                sde_indices=sde_indices,
                requirements=requirements,
                sampling_overrides=sampling_overrides,
            ),
            validate_sampler_outputs=partial(
                validate_sampler_outputs_against_contract,
                requirements=requirements,
                allow_replay=bool(validation_config.get("allow_replay", False)),
                assert_step_alignment=bool(validation_config.get("assert_step_alignment", True)),
                mode_label=str(validation_config.get("mode_label", "trajectory")),
            ),
        )
        sample_t1 = _time.perf_counter()
        sampler_outputs = sampled_rollout.sampler_outputs
        train_prompts = sampled_rollout.train_prompts
        train_prompt_ids = sampled_rollout.train_prompt_ids
        sample_ids = sampled_rollout.sample_ids
        group_ids = sampled_rollout.group_ids
        prompt_metadata = sampled_rollout.prompt_metadata

        reward_t0 = _time.perf_counter()
        rewards, reward_components = self._compute_rewards_only(
            sampler_outputs=sampler_outputs,
            prompts=train_prompts if train_prompts else prompts,
            prompt_ids=train_prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )
        advantages = compute_advantages(
            algorithm=self.algorithm,
            rewards=rewards,
            group_ids=group_ids,
            reward_components=reward_components,
            reward_component_weights=self.reward_component_weights,
        )
        reward_t1 = _time.perf_counter()

        metadata: Dict[str, Any] = {}
        if collect_media_preview:
            reward_prompts = self._build_reward_prompts(
                prompts=train_prompts if train_prompts else prompts,
                sample_count=int(rewards.shape[0]),
            )
            media_preview = self._build_wandb_media_preview(
                sampler_outputs=sampler_outputs,
                reward_prompts=reward_prompts,
                rewards=rewards,
                max_items=media_max_items,
            )
            if media_preview is not None:
                metadata["wandb_media_preview"] = media_preview

        assemble_t0 = _time.perf_counter()
        assembled_batch = self.algorithm.assemble_training_batch(
            num_inference_steps=request_num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )
        training_batch = self._attach_batch_identities(
            batch=assembled_batch,
            prompt_ids=train_prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
        )
        assemble_t1 = _time.perf_counter()
        logger.debug(
            "[TIMING] rollout_workflow rollout=%s: sample=%.2fs reward_advantage=%.2fs assemble=%.2fs total=%.2fs",
            rollout_id,
            sample_t1 - sample_t0,
            reward_t1 - reward_t0,
            assemble_t1 - assemble_t0,
            assemble_t1 - sample_t0,
        )

        if debug_trace is not None:
            reward_prompts = self._build_reward_prompts(
                prompts=train_prompts if train_prompts else prompts,
                sample_count=int(rewards.shape[0]),
            )
            debug_trace.update(
                {
                    "rollout_id": int(rollout_id),
                    "debug_mode": str(getattr(self.args.debug, "debug_mode", "none")),
                    "prompts": list(prompts),
                    "train_prompts": list(train_prompts if train_prompts else prompts),
                    "prompt_ids": list(train_prompt_ids or []),
                    "sample_ids": list(sample_ids or []),
                    "group_ids": list(group_ids or []),
                    "reward_prompts": reward_prompts,
                    "sde_indices": sorted(int(v) for v in (sde_indices or [])),
                    "sampler_outputs": sampler_outputs,
                    "rewards": rewards,
                    "advantages": advantages,
                    "reward_components": reward_components,
                }
            )

        return training_batch, metadata or None

    def _load_prompt_batch(self) -> Dict[str, Any]:
        batch = self.load_prompt_batch_fn()
        if not isinstance(batch, dict):
            raise TypeError(
                "Rollout workflow prompt batch callback must return Dict[str, Any], "
                f"got {type(batch).__name__}."
            )
        return batch

    def _build_reward_prompts(
        self,
        *,
        prompts: List[str],
        sample_count: int,
    ) -> List[str]:
        candidate = list(prompts)
        if not candidate:
            return []
        expanded: List[str] = []
        while len(expanded) < sample_count:
            expanded.extend(candidate)
        return expanded[:sample_count]

    def _build_wandb_media_preview(
        self,
        *,
        sampler_outputs: List[Any],
        reward_prompts: List[str],
        rewards: torch.Tensor,
        max_items: int,
    ) -> Optional[Dict[str, Any]]:
        limit = max(1, int(max_items))
        rewards_flat: List[float] = []
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_flat = [float(v) for v in rewards.detach().cpu().reshape(-1).tolist()]

        images: List[Any] = []
        prompts: List[str] = []
        reward_values: List[float] = []
        global_sample_idx = 0

        for output in sampler_outputs:
            batch_size = int(getattr(output, "batch_size", 0) or 0)
            decoded_images = list(getattr(output, "decoded_images", None) or [])
            for image_idx, image in enumerate(decoded_images):
                if len(images) >= limit:
                    break
                if not hasattr(image, "save"):
                    continue
                sample_idx = global_sample_idx + image_idx
                images.append(image)
                prompt = reward_prompts[sample_idx] if sample_idx < len(reward_prompts) else ""
                prompts.append(str(prompt))
                reward_val = rewards_flat[sample_idx] if sample_idx < len(rewards_flat) else 0.0
                reward_values.append(float(reward_val))
            if len(images) >= limit:
                break
            global_sample_idx += batch_size

        if not images:
            return None

        return {
            "images": images,
            "prompts": prompts,
            "rewards": reward_values,
        }

    def _compute_rewards_only(
        self,
        *,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        group_ids: Optional[List[str]] = None,
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        samples_per_prompt_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        samples_per_prompt = int(
            samples_per_prompt_override
            if samples_per_prompt_override is not None
            else getattr(self.algorithm, "samples_per_prompt", 1)
        )

        return score_reward_stage(
            reward_scoring_mode=self.reward_scoring_mode,
            reward_service=self.reward_service,
            samples_per_prompt=samples_per_prompt,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )

    def _attach_batch_identities(
        self,
        *,
        batch: TrainingBatch,
        prompt_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        group_ids: Optional[List[str]] = None,
    ) -> TrainingBatch:
        batch_size = int(getattr(batch, "batch_size", 0))
        if batch_size <= 0:
            return batch

        resolved_prompt_ids = prompt_ids if prompt_ids is not None else getattr(batch, "prompt_ids", None)
        if resolved_prompt_ids is None or len(resolved_prompt_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit sample-aligned prompt_ids. "
                f"Got batch_size={batch_size}, prompt_ids_len="
                f"{len(resolved_prompt_ids) if resolved_prompt_ids is not None else None}."
            )
        batch.prompt_ids = list(resolved_prompt_ids)

        resolved_sample_ids = sample_ids if sample_ids is not None else getattr(batch, "sample_ids", None)
        if resolved_sample_ids is None or len(resolved_sample_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit sample_ids aligned to the reward batch. "
                f"Got batch_size={batch_size}, sample_ids_len="
                f"{len(resolved_sample_ids) if resolved_sample_ids is not None else None}."
            )
        batch.sample_ids = list(resolved_sample_ids)

        resolved_group_ids = group_ids if group_ids is not None else getattr(batch, "group_ids", None)
        if resolved_group_ids is None or len(resolved_group_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit group_ids aligned to the reward batch. "
                f"Got batch_size={batch_size}, group_ids_len="
                f"{len(resolved_group_ids) if resolved_group_ids is not None else None}."
            )
        batch.group_ids = list(resolved_group_ids)

        return batch


# =========================================================================
# Sampling stage
# =========================================================================


def distributed_sample(
    *,
    actor_group: Any,
    request: RolloutRequest,
) -> List[RolloutOutput]:
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
            meta = getattr(out, "metadata", {}) or {}
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
                resolved_steps = out.resolved_step_indices
                if int(resolved_steps.shape[0]) != int(out.timesteps.shape[0]):
                    raise ValueError(
                        f"step/timestep length mismatch: step_indices={resolved_steps.shape[0]}, "
                        f"timesteps={out.timesteps.shape[0]}"
                    )
        except Exception as exc:
            meta = getattr(out, "metadata", {}) or {}
            generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
            capabilities = meta.get("engine_capabilities") if isinstance(meta, dict) else None
            traj_shape = tuple(out.trajectories.shape) if getattr(out, "trajectories", None) is not None else None
            latents_shape = tuple(out.latents.shape) if getattr(out, "latents", None) is not None else None
            steps_shape = tuple(out.resolved_step_indices.shape) if hasattr(out, "resolved_step_indices") else None
            hint = ""
            if generator_type == "sglang":
                hint = (
                    f" {generator_type} currently may omit rollout log_probs; "
                    "enable replay_log_probs and ensure prompt text inputs are present."
                )
            raise RuntimeError(
                f"Sampler output contract validation failed in {mode_label} path at index={idx}: {exc}.{hint} "
                f"capabilities={capabilities}, latents_shape={latents_shape}, "
                f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
            ) from exc


# =========================================================================
# Advantage stage
# =========================================================================


def compute_advantages(
    *,
    algorithm: Any,
    rewards: torch.Tensor,
    group_ids: Optional[List[str]] = None,
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Delegate reward-component-aware advantage semantics to the algorithm."""
    return algorithm.compute_advantages_with_components(
        rewards=rewards,
        group_ids=group_ids,
        reward_components=reward_components,
        reward_component_weights=reward_component_weights,
    )


__all__ = [
    "RolloutWorkflow",
    "distributed_sample",
    "validate_sampler_outputs_against_contract",
    "compute_advantages",
]
