"""
MixGRPO Algorithm Implementation.

Mixed SDE/ODE GRPO with configurable ratio.
"""
from typing import Any, Dict, Optional, Set

from .base import SamplingRequirements
from .grpo import GRPOAlgorithm


class MixGRPOAlgorithm(GRPOAlgorithm):
    """
    Mixed SDE/ODE GRPO Algorithm.

    Uses a mix of SDE steps (for training) and ODE steps (for speed).
    Only computes loss on SDE steps.

    Supports two independent mechanisms:
    - window_scheduler: Dynamic SDE window (controls which steps use SDE sampling)
    - window_training: Only train on window timesteps (controls which steps compute loss)
    """

    def __init__(
        self,
        sde_ratio: float = 0.5,
        window_training: bool = False,
        **kwargs,
    ):
        """
        Initialize MixGRPO.

        Args:
            sde_ratio: Ratio of steps to use SDE (0-1)
            window_training: If True, only compute loss on SDE window timesteps.
                When enabled, get_training_indices returns SDE indices.
                When disabled, trains on all timesteps (standard behavior).
            **kwargs: Additional GRPO arguments
        """
        super().__init__(**kwargs)
        self.sde_ratio = sde_ratio
        self.window_training = window_training

        # Current SDE indices (updated by scheduler)
        self._current_sde_indices: Optional[Set[int]] = None

    @classmethod
    def from_args(cls, args: Any) -> "MixGRPOAlgorithm":
        """Construct MixGRPO algorithm from runtime args."""
        kwargs: Dict[str, Any] = cls._grpo_kwargs_from_args(args)
        kwargs.update(
            {
                "sde_ratio": getattr(args, "sde_ratio", 1.0),
                "window_training": getattr(args, "window_training", False),
            }
        )
        kwargs.update(cls._algorithm_kwargs_from_args(args))
        return cls(**kwargs)

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return MixGRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            sde_ratio=self.sde_ratio,
        )

    def get_sde_indices(self, num_steps: int) -> Set[int]:
        """
        Compute which timestep indices should use SDE based on sde_ratio.

        Uses first N steps as SDE where N = ceil(num_steps * sde_ratio).
        Early steps (high noise) benefit most from SDE sampling.

        Args:
            num_steps: Total number of inference steps

        Returns:
            Set of timestep indices that should use SDE
        """
        if self.sde_ratio >= 1.0:
            return set(range(num_steps))
        elif self.sde_ratio <= 0.0:
            # At least 1 SDE step to have trainable loss
            return {0}

        num_sde_steps = max(1, int(num_steps * self.sde_ratio + 0.5))
        return set(range(num_sde_steps))

    def set_sde_indices(self, sde_indices: Set[int]) -> None:
        """
        Set the current SDE indices from scheduler.

        This is called by the rollout manager to update the algorithm's
        knowledge of which timesteps use SDE for the current iteration.

        Args:
            sde_indices: Set of timestep indices using SDE
        """
        self._current_sde_indices = sde_indices

    def get_training_indices(self, num_steps: int) -> Set[int]:
        """
        Get the timestep indices to train on.

        When window_training is enabled, only train on SDE timesteps.
        Otherwise, train on all timesteps.

        Args:
            num_steps: Total number of timesteps

        Returns:
            Set of timestep indices to compute loss for
        """
        if self.window_training:
            # Only train on current SDE window
            if self._current_sde_indices is not None:
                return self._current_sde_indices
            else:
                return self.get_sde_indices(num_steps)
        else:
            # Train on all timesteps (standard behavior)
            return set(range(num_steps))
