"""Ray-agnostic rollout pipeline executor used by RolloutManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from diffusionrl.runtime.pipeline.advantage_stage import compute_advantages
from diffusionrl.runtime.pipeline.assemble_stage import (
    assemble_backward_training_batch,
    assemble_forward_training_batch,
)
from diffusionrl.runtime.pipeline.reward_stage import compute_rewards
from diffusionrl.runtime.pipeline.sampling_stage import (
    distributed_sample,
    expand_batch_for_sampling,
)
from diffusionrl.types.training_batch import TrainingBatch

logger = logging.getLogger(__name__)


class RolloutPipelineExecutor:
    """Pure rollout execution logic independent from Ray actor lifecycle."""

    def __init__(self, args) -> None:
        self.args = args
        self._warned_ignored_prompt_embeddings = False

    def prepare_batch(self, *, data_source: Any) -> Dict[str, Any]:
        """Fetch one prompt batch from data source."""
        if data_source is not None:
            batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
            samples = data_source.get_samples(batch_size)
            if isinstance(samples, dict):
                return samples
            raise TypeError(
                "DataSource.get_samples() must return Dict[str, Any] with at least 'prompts'. "
                f"Got {type(samples).__name__}."
            )

        default_prompts = [
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A mountain landscape with snow",
            "A futuristic city at night",
        ]
        batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
        return {"prompts": default_prompts[:batch_size]}

    def sample(
        self,
        *,
        actor_group: Any,
        batch: Dict[str, Any],
        sde_indices: Optional[Set[int]],
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], List[str], List[str]]:
        """Run distributed sampling with prompt-major K expansion."""
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "Rollout sampling requires non-empty text prompts in batch['prompts']. "
                "Prompt-embedding-only batches are no longer supported."
            )

        embedding_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "text_ids",
            "image_ids",
            "encoder_attention_mask",
        )
        if (
            not self._warned_ignored_prompt_embeddings
            and any(key in batch for key in embedding_keys)
        ):
            logger.warning(
                "Rollout sampling now uses prompt-only input; batch embedding fields are ignored."
            )
            self._warned_ignored_prompt_embeddings = True

        overrides = dict(sampling_overrides or {})
        num_samples_per_prompt = int(
            overrides.pop(
                "num_samples_per_prompt",
                getattr(self.args, "num_samples_per_prompt", 1),
            )
        )
        init_same_noise = bool(
            overrides.pop(
                "init_same_noise",
                getattr(self.args, "init_same_noise", False),
            )
        )
        num_inference_steps = int(
            overrides.pop(
                "num_inference_steps",
                getattr(self.args, "num_inference_steps", 50),
            )
        )
        guidance_scale = float(
            overrides.pop(
                "guidance_scale",
                getattr(self.args, "guidance_scale", 7.5),
            )
        )
        height = int(overrides.pop("height", getattr(self.args, "height", 256)))
        width = int(overrides.pop("width", getattr(self.args, "width", 256)))
        num_frames = int(
            overrides.pop(
                "num_frames",
                getattr(self.args, "num_frames", 16),
            )
        )

        sampling_batch, train_prompts = expand_batch_for_sampling(
            {"prompts": prompts, "metadata": batch.get("metadata"), "latents": batch.get("latents")},
            num_samples_per_prompt=num_samples_per_prompt,
        )
        sampler_outputs = distributed_sample(
            actor_group=actor_group,
            batch=sampling_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            init_same_noise=init_same_noise,
            num_samples_per_prompt=num_samples_per_prompt,
            sde_indices=sde_indices,
            extra_generate_kwargs=overrides,
        )
        return sampler_outputs, (train_prompts if train_prompts is not None else prompts), prompts

    def validate_sampler_outputs(
        self,
        *,
        sampler_outputs: List[Any],
        requirements: Any,
        allow_replay: bool,
        assert_step_alignment: bool,
        mode_label: str,
    ) -> None:
        """Validate sampler outputs against algorithm requirements."""
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
            except Exception as e:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                capabilities = meta.get("engine_capabilities") if isinstance(meta, dict) else None
                traj_shape = tuple(out.trajectories.shape) if getattr(out, "trajectories", None) is not None else None
                latents_shape = tuple(out.latents.shape) if getattr(out, "latents", None) is not None else None
                steps_shape = tuple(out.resolved_step_indices.shape) if hasattr(out, "resolved_step_indices") else None
                hint = ""
                if generator_type in {"fastvideo", "sglang"}:
                    hint = (
                        f" {generator_type} currently may omit rollout log_probs; "
                        "enable replay_log_probs and ensure prompt text inputs are present."
                    )
                raise RuntimeError(
                    f"Sampler output contract validation failed in {mode_label} path at index={idx}: {e}.{hint} "
                    f"capabilities={capabilities}, latents_shape={latents_shape}, "
                    f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
                ) from e

    def compute_reward_and_advantage(
        self,
        *,
        algorithm: Any,
        reward_service: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards and advantages from sampler outputs."""
        rewards, reward_components = self.compute_rewards_only(
            reward_service=reward_service,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
        )

        advantages = compute_advantages(
            algorithm=algorithm,
            num_samples_per_prompt=int(getattr(self.args, "num_samples_per_prompt", 1)),
            reward_mix_mode=str(getattr(self.args, "reward_mix_mode", "reward_aggr")),
            rewards=rewards,
            prompts=prompts,
            reward_components=reward_components,
            reward_workers=getattr(reward_service, "workers", None) if reward_service is not None else None,
        )
        return rewards, advantages, reward_components

    def compute_rewards_only(
        self,
        *,
        reward_service: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        reward_path_override: Optional[str] = None,
        num_samples_per_prompt_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards only (shared by default path and custom pipelines)."""
        num_samples_per_prompt = int(
            num_samples_per_prompt_override
            if num_samples_per_prompt_override is not None
            else getattr(self.args, "num_samples_per_prompt", 1)
        )
        reward_path = str(
            reward_path_override
            if reward_path_override is not None
            else getattr(self.args, "reward_path", "")
        )

        if reward_service is None:
            raise RuntimeError("RewardService is not initialized.")
        return compute_rewards(
            reward_service=reward_service,
            reward_path=reward_path,
            num_samples_per_prompt=num_samples_per_prompt,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
        )

    def assemble_training_batch(
        self,
        *,
        algorithm: Any,
        requirements: Any,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]],
    ) -> TrainingBatch:
        """Assemble typed training batch for backward (GRPO) or forward (NFT) paths."""
        if bool(requirements.is_forward_process):
            return assemble_forward_training_batch(
                sampler_outputs=sampler_outputs,
                rewards=rewards,
                advantages=advantages,
                prompts=prompts,
            )

        return assemble_backward_training_batch(
            algorithm=algorithm,
            num_inference_steps=int(self.args.num_inference_steps),
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=prompts,
            sde_indices=sde_indices,
        )

    def eval_batch(
        self,
        *,
        rollout_id: int,
        data_source: Any,
        actor_group: Any,
        reward_service: Any,
    ) -> Dict[str, Any]:
        """Run evaluation sampling and reward aggregation."""
        if data_source is not None and hasattr(data_source, "get_eval_samples"):
            prompts = data_source.get_eval_samples(self.args.eval_batch_size)
        else:
            prompts = self.prepare_batch(data_source=data_source).get("prompts", [])[: self.args.eval_batch_size]

        outputs = distributed_sample(
            actor_group=actor_group,
            batch={"prompts": prompts},
            num_inference_steps=int(self.args.num_inference_steps),
            guidance_scale=float(self.args.guidance_scale),
            height=int(self.args.height),
            width=int(self.args.width),
            num_frames=int(self.args.num_frames),
            init_same_noise=bool(getattr(self.args, "init_same_noise", False)),
            num_samples_per_prompt=int(getattr(self.args, "num_samples_per_prompt", 1)),
            sde_indices=None,
        )
        rewards, _ = self.compute_rewards_only(
            reward_service=reward_service,
            sampler_outputs=outputs,
            prompts=prompts,
        )

        return {
            "rollout_id": rollout_id,
            "num_samples": len(prompts),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "prompts": prompts,
        }
