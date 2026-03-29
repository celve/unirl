"""Minimal algorithm plugin example using the current algorithm API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import BaseAlgorithm, EMASpec, SamplingRequirements
from diffusionrl.algorithms.grpo import GRPOAlgorithm
from diffusionrl.types import PromptEmbeddings, TimestepData
from diffusionrl.types.sampling import RolloutRequest


class MinimalAlgorithm(BaseAlgorithm):
    """Small GRPO-style plugin example with a placeholder loss."""

    @classmethod
    def from_config(cls, config: dict) -> "MinimalAlgorithm":
        extra = cls.resolve_config_kwargs(config)
        known_keys = {
            "skip_last_timestep",
            "skip_initial_timesteps",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys)
        if unknown:
            raise ValueError(
                "algorithm.algorithm_kwargs contains unsupported keys for MinimalAlgorithm: "
                f"{unknown}."
            )

        return cls(
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            component_mix_stage=str(config.get("component_mix_stage", "reward")),
            adv_normalization=str(config.get("adv_normalization", "group")),
            samples_per_prompt=int(config.get("samples_per_prompt", 1)),
            num_inference_steps=int(config.get("num_inference_steps", 0)),
            eval_ema_decay=float(config.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(config.get("eval_ema_update_interval", 1)),
            epsilon=float(config.get("adv_norm_eps", 1e-8)),
            clip_max=config.get("adv_clip_abs", 5.0),
            use_global_std=bool(config.get("use_global_std", False)),
            trimmed_ratio=float(config.get("trimmed_ratio", 0.0)),
        )

    def __init__(
        self,
        *,
        skip_last_timestep: bool = False,
        skip_initial_timesteps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.skip_last_timestep = bool(skip_last_timestep)
        self.skip_initial_timesteps = int(skip_initial_timesteps)

    @classmethod
    def from_args(cls, args: Any) -> "MinimalAlgorithm":
        from diffusionrl.algorithms.construction import build_algorithm_config

        return cls.from_config(build_algorithm_config(args))

    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
        )

    def compute_advantages_with_components(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[list[str]] = None,
        reward_components: Optional[Dict[str, list[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        return GRPOAlgorithm.compute_advantages_with_components(
            self,
            rewards=rewards,
            group_ids=group_ids,
            reward_components=reward_components,
            reward_component_weights=reward_component_weights,
        )

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(enable_eval_ema=False)

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Set[int]:
        del current_step
        if self.num_inference_steps < 1:
            raise ValueError(
                f"{type(self).__name__}.resolve_rollout_sde_indices requires "
                f"num_inference_steps >= 1, got {self.num_inference_steps}."
            )
        return set(range(self.num_inference_steps))

    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        return {
            "allow_replay": bool(getattr(args.sampling, "replay_log_probs", False)),
            "assert_step_alignment": True,
            "mode_label": "trajectory",
        }

    def assemble_training_batch(
        self,
        *,
        request: RolloutRequest,
        sampler_outputs: list[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        return GRPOAlgorithm.assemble_training_batch(
            self,
            request=request,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            sde_indices=sde_indices,
        )

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        del num_steps
        result = set(sde_indices)
        if self.skip_last_timestep and result:
            result.discard(max(result))
        if self.skip_initial_timesteps > 0:
            result = {i for i in result if i >= self.skip_initial_timesteps}
        return result

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        del current_step, kwargs
        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        step_indices = batch.resolved_step_indices[:-1]
        step_labels = set(int(v) for v in step_indices.tolist())
        if not step_labels:
            return tuple()

        filtered_steps = self.get_filtered_training_indices(
            set(int(i) for i in batch.sde_indices),
            len(step_labels),
        )
        missing_steps = sorted(
            int(i) for i in filtered_steps if int(i) not in step_labels
        )
        if missing_steps:
            raise ValueError(
                f"{type(self).__name__}.resolve_training_timesteps selected steps "
                f"not present in batch: missing={missing_steps}, "
                f"available={sorted(step_labels)}"
            )
        if not filtered_steps:
            return tuple()

        selected_positions = [
            pos
            for pos, step_label in enumerate(step_indices.tolist())
            if int(step_label) in filtered_steps
        ]
        return batch.timesteps[selected_positions]

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Placeholder single-timestep loss for custom plugin experiments."""
        del model, advantages, embeddings, kwargs
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder_loss": loss.detach()}

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        timesteps: Any,
        guidance_scale: float = 3.5,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        del kwargs
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        model.train()
        if torch.is_tensor(timesteps):
            candidate_timesteps = timesteps.detach().flatten()
        else:
            candidate_timesteps = torch.as_tensor(
                list(timesteps),
                device=batch.timesteps.device,
                dtype=batch.timesteps.dtype,
            )
        num_timesteps_per_sample = int(candidate_timesteps.numel())
        if num_timesteps_per_sample == 0:
            return 0.0, {}, 0, False

        total_loss_accum = 0.0
        has_backward = False

        for timestep_value in candidate_timesteps:
            timestep_data = batch.get_timestep_data_by_timestep(timestep_value)
            loss, _ = self.compute_loss(
                model=model,
                timestep_data=timestep_data,
                advantages=batch.advantages,
                embeddings=batch.embeddings,
            )
            scaled_loss = loss * (loss_scale / num_timesteps_per_sample)
            scaled_loss.backward()
            total_loss_accum += scaled_loss.detach().item()
            has_backward = True

        return (
            total_loss_accum,
            {"placeholder": 1.0},
            num_timesteps_per_sample,
            has_backward,
        )
