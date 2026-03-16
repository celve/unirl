"""
Minimal algorithm plugin example — algorithm-owned loss in one file.

Copy this file and edit the class body to implement a custom algorithm.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import BaseAlgorithm, EMASpec, SamplingRequirements
from diffusionrl.algorithms.grpo import GRPOAlgorithm
from diffusionrl.types import PromptEmbeddings, TimestepData


class _MinimalLoss:
    """Tiny private loss class used by MinimalAlgorithm."""

    def __init__(self, algorithm: "MinimalAlgorithm") -> None:
        self.algorithm = algorithm

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del model, advantages, embeddings, kwargs
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder": True}


class MinimalAlgorithm(BaseAlgorithm):
    """Small unified algorithm plugin example.

    This class demonstrates the single-file algorithm pattern:
    - private `_MinimalLoss`
    - `compute_loss()` as the single public loss entrypoint
    - algorithm-owned training step
    """
    _loss_cls = _MinimalLoss

    @classmethod
    def from_config(cls, config: dict) -> "MinimalAlgorithm":
        extra = config.get("algorithm_kwargs") or {}
        return cls(
            sde_ratio=float(extra.get("sde_ratio", config.get("sde_ratio", 1.0))),
            train_only_sde_steps=bool(extra.get("train_only_sde_steps", False)),
            skip_last_timestep=bool(extra.get("skip_last_timestep", config.get("skip_last_timestep", False))),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", config.get("skip_initial_timesteps", 0))),
        )

    def __init__(
        self,
        *,
        sde_ratio: float = 1.0,
        train_only_sde_steps: bool = False,
        skip_last_timestep: bool = False,
        skip_initial_timesteps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sde_ratio = float(sde_ratio)
        self.train_only_sde_steps = bool(train_only_sde_steps)
        self.skip_last_timestep = bool(skip_last_timestep)
        self.skip_initial_timesteps = int(skip_initial_timesteps)
        self._current_sde_indices: Optional[Set[int]] = None
        self._forward_plugin = None
        self.model_type = "default"

    @classmethod
    def from_args(cls, args: Any) -> "MinimalAlgorithm":
        from diffusionrl.config.build_domain_args import build_algorithm_config

        return cls.from_config(build_algorithm_config(args))

    def get_sampling_requirements(self) -> SamplingRequirements:
        return self._build_sampling_requirements(extras={"sde_ratio": self.sde_ratio})

    def compute_advantages_with_components(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[list[str]] = None,
        component_mix_stage: str = "reward",
        reward_components: Optional[Dict[str, list[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        return GRPOAlgorithm.compute_advantages_with_components(
            self,
            rewards=rewards,
            group_ids=group_ids,
            component_mix_stage=component_mix_stage,
            reward_components=reward_components,
            reward_component_weights=reward_component_weights,
        )

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(enable_eval_ema=False)

    def resolve_rollout_sde_indices(
        self,
        *,
        timestep_scheduler: Optional[Any],
        current_step: int,
    ) -> Optional[Set[int]]:
        if timestep_scheduler is None:
            return None
        sde_indices = set(int(i) for i in timestep_scheduler.get_sde_indices(current_step))
        self.set_sde_indices(sde_indices)
        return sde_indices

    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        return {
            "allow_replay": bool(getattr(args.sampling, "replay_log_probs", False)),
            "assert_step_alignment": True,
            "mode_label": "trajectory",
        }

    def assemble_training_batch(
        self,
        *,
        num_inference_steps: int,
        sampler_outputs: list[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: list[str],
        sde_indices: Optional[Set[int]] = None,
    ):
        return GRPOAlgorithm.assemble_training_batch(
            self,
            num_inference_steps=num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=prompts,
            sde_indices=sde_indices,
        )

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single loss entrypoint for debugging and training."""
        if self.loss_fn is None:
            raise RuntimeError(f"{type(self).__name__} loss_fn is not initialized.")
        return self.loss_fn.compute_loss(
            model=model,
            timestep_data=timestep_data,
            advantages=advantages,
            embeddings=embeddings,
            **kwargs,
        )

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        gradient_accumulation_batch_size: int,
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> tuple:
        del guidance_scale, kwargs
        from diffusionrl.runtime.training.update_schedule import resolve_gradient_accumulation_plan
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        mini_batches, actual_mini_batches = resolve_gradient_accumulation_plan(
            batch_size=batch.batch_size,
            gradient_accumulation_batch_size=gradient_accumulation_batch_size,
        )
        num_mini_batches = len(mini_batches)
        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        valid_step_indices = sorted(int(i) for i in batch.sde_indices if int(i) in available_steps)
        if not valid_step_indices:
            return 0.0, {}, 0, actual_mini_batches, False

        total_loss_accum = 0.0
        has_backward = False

        for start, end in mini_batches:
            mini_batch = batch.slice(start, end)
            for t_idx in valid_step_indices:
                timestep_data = mini_batch.get_timestep_data_by_step(t_idx)
                loss, _ = self.compute_loss(
                    model=model,
                    timestep_data=timestep_data,
                    advantages=mini_batch.advantages,
                    embeddings=mini_batch.embeddings,
                )
                scaled_loss = loss / (num_mini_batches * len(valid_step_indices))
                scaled_loss.backward()
                total_loss_accum += scaled_loss.detach().item()
                has_backward = True

        return total_loss_accum, {"placeholder": 1.0}, len(valid_step_indices), actual_mini_batches, has_backward

    def set_sde_indices(self, sde_indices: Set[int]) -> None:
        """Optional callback used by RolloutManager when scheduler updates."""
        self._current_sde_indices = set(int(i) for i in sde_indices)

    def get_training_indices(self, num_steps: int) -> Set[int]:
        """Optional hook to constrain which timesteps are optimized."""
        if not self.train_only_sde_steps:
            return set(range(num_steps))

        if self._current_sde_indices is not None:
            return set(int(i) for i in self._current_sde_indices)

        # Fallback when scheduler callback has not run yet.
        num_sde_steps = max(1, int(num_steps * self.sde_ratio + 0.5))
        return set(range(num_sde_steps))

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
