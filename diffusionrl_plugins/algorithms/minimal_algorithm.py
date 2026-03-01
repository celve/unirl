"""
Minimal algorithm plugin example.

Copy this file and edit the class body to implement a custom algorithm.
Always import shared data types from `diffusionrl.types` in new code.
"""

from __future__ import annotations

from typing import Any, Optional, Set

from diffusionrl.algorithms.base import BaseAlgorithm, SamplingRequirements


class MinimalAlgorithm(BaseAlgorithm):
    """Small, rollout-centric algorithm plugin example.

    This class demonstrates what the current framework actually calls:
    - from_args(): construct from CLI/runtime config
    - get_sampling_requirements(): tell sampler what to return
    - compute_advantages(): reward -> advantage transformation
    - optional timestep filtering hooks for backward training batches

    In the default training path, gradients are computed by loss_fn
    (configured via --loss-type/--loss-path), not by algorithm.compute_loss().
    """

    def __init__(
        self,
        *,
        sde_ratio: float = 1.0,
        train_only_sde_steps: bool = False,
        ignore_last: bool = False,
        frozen_init_timesteps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sde_ratio = float(sde_ratio)
        self.train_only_sde_steps = bool(train_only_sde_steps)
        self.ignore_last = bool(ignore_last)
        self.frozen_init_timesteps = int(frozen_init_timesteps)
        self._current_sde_indices: Optional[Set[int]] = None

    @classmethod
    def from_args(cls, args: Any) -> "MinimalAlgorithm":
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "sde_ratio": getattr(args, "sde_ratio", 1.0),
                "train_only_sde_steps": getattr(args, "window_training", False),
                "ignore_last": getattr(args, "ignore_last", False),
                "frozen_init_timesteps": getattr(args, "frozen_init_timesteps", 0),
            }
        )
        return cls(**kwargs)

    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
            extras={"sde_ratio": self.sde_ratio},
        )

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
        # Reuse common ignore_last/frozen_init_timesteps behavior from BaseAlgorithm.
        return super().get_filtered_training_indices(sde_indices=sde_indices, num_steps=num_steps)
