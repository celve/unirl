"""
Timestep schedulers for MixGRPO training.

This module implements sliding window timestep scheduling, which determines
which timesteps use SDE (stochastic) vs ODE (deterministic) sampling.

Key concepts:
- In MixGRPO, only a subset of timesteps use SDE sampling
- SDE timesteps compute log_prob and contribute to policy gradient loss
- ODE timesteps are faster but don't contribute to training

Based on: MixGRPO/fastvideo/utils/grpo_states.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Dict, Any, Type

import numpy as np


Strategy = Literal["all", "progressive", "random", "decay", "exp_decay"]


@dataclass
class WindowConfig:
    """
    Configuration for sliding window scheduler.

    Attributes:
        strategy: Scheduling strategy
            - "all": All timesteps use SDE (standard GRPO)
            - "progressive": Slide window from early to late timesteps
            - "random": Random timestep selection each iteration
            - "decay": Linear decay of iterations per group
            - "exp_decay": Exponential decay of iterations per group
        group_size: Number of timesteps in each group (window size)
        iters_per_group: Number of iterations to train on each group
        overlap: Whether to use overlapping windows
        overlap_step: Step size for overlap (1 = maximum overlap)
        roll_back: Whether to roll back to start when reaching end
        max_iters_per_group: Maximum iterations (for decay strategy)
        min_iters_per_group: Minimum iterations (for decay strategy)
        exp_decay_threshold: Threshold timestep for exp_decay
        exp_decay_k: Decay rate for exp_decay
    """
    strategy: Strategy = "all"
    group_size: int = 4
    iters_per_group: int = 25
    overlap: bool = False
    overlap_step: int = 1
    roll_back: bool = False
    max_iters_per_group: Optional[int] = None
    min_iters_per_group: Optional[int] = None
    exp_decay_threshold: int = 13
    exp_decay_k: float = 0.1

    def __post_init__(self):
        if self.strategy == "decay":
            if self.max_iters_per_group is None:
                self.max_iters_per_group = self.iters_per_group
            if self.min_iters_per_group is None:
                self.min_iters_per_group = max(1, self.iters_per_group // 4)


class TimestepScheduler(ABC):
    """
    Abstract base class for timestep schedulers.

    Schedulers determine which timesteps use SDE sampling (and thus
    contribute to the policy gradient loss).
    """

    def __init__(self, num_timesteps: int):
        """
        Initialize scheduler.

        Args:
            num_timesteps: Total number of denoising timesteps
        """
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """
        Get the set of timestep indices that should use SDE sampling.

        Args:
            step: Current training step (for dynamic schedulers)

        Returns:
            Set of timestep indices that should use SDE
        """
        pass

    @abstractmethod
    def update(self, step: int) -> None:
        """
        Update scheduler state after a training step.

        Args:
            step: Current training step
        """
        pass

    def get_current_timesteps(self) -> List[int]:
        """Get list of current SDE timesteps (sorted)."""
        return sorted(self.get_sde_indices())

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {"num_timesteps": self.num_timesteps}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.num_timesteps = state_dict.get("num_timesteps", self.num_timesteps)


class AllSDEScheduler(TimestepScheduler):
    """
    All SDE scheduler (standard GRPO) with optional timestep_fraction (DanceGRPO).

    When timestep_fraction=1.0 (default): All timesteps use SDE sampling.
    When timestep_fraction<1.0: Only the first fraction of timesteps use SDE.

    This implements DanceGRPO's timestep_fraction parameter which restricts
    training to early timesteps (e.g., timestep_fraction=0.6 trains on first 60%
    of timesteps only).
    """

    def __init__(self, num_timesteps: int, timestep_fraction: float = 1.0):
        """
        Initialize AllSDE scheduler.

        Args:
            num_timesteps: Total number of denoising timesteps
            timestep_fraction: Fraction of timesteps to train (1.0 = all, 0.6 = first 60%)
        """
        super().__init__(num_timesteps)
        self.timestep_fraction = timestep_fraction
        self._effective_timesteps = int(num_timesteps * timestep_fraction)

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Return first fraction of timesteps."""
        return set(range(self._effective_timesteps))

    def update(self, step: int) -> None:
        """No-op for all SDE scheduler."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            "num_timesteps": self.num_timesteps,
            "timestep_fraction": self.timestep_fraction,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        super().load_state_dict(state_dict)
        self.timestep_fraction = state_dict.get("timestep_fraction", 1.0)
        self._effective_timesteps = int(self.num_timesteps * self.timestep_fraction)


class WindowScheduler(TimestepScheduler):
    """
    Sliding window scheduler (MixGRPO).

    Only a subset of timesteps use SDE sampling at any given time.
    The window slides through the timesteps as training progresses.

    This implements the progressive training strategy from MixGRPO,
    where training focuses on different parts of the denoising trajectory
    over time.

    Example:
        # Window of 4 timesteps, 25 iterations per position
        scheduler = WindowScheduler(
            num_timesteps=50,
            config=WindowConfig(
                strategy="progressive",
                group_size=4,
                iters_per_group=25,
            )
        )

        for step in range(1000):
            sde_indices = scheduler.get_sde_indices(step)
            # Train on these timesteps...
            scheduler.update(step)
    """

    def __init__(self, num_timesteps: int, config: WindowConfig):
        """
        Initialize window scheduler.

        Args:
            num_timesteps: Total number of denoising timesteps
            config: Window configuration
        """
        super().__init__(num_timesteps)
        self.config = config

        # Current state
        self.cur_timestep = 0  # Start of current window
        self.cur_iter_in_group = 0  # Current iteration within group
        self.init_timestep = 0  # For roll_back

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Get current window's timestep indices."""
        if self.config.strategy == "all":
            return set(range(self.num_timesteps))

        # Return current window
        end = min(self.cur_timestep + self.config.group_size, self.num_timesteps)
        return set(range(self.cur_timestep, end))

    def update(self, step: int) -> None:
        """Update window position based on strategy."""
        if self.config.strategy == "all":
            return

        if self.config.strategy == "progressive":
            self._update_progressive()
        elif self.config.strategy == "random":
            self._update_random(step)
        elif self.config.strategy == "decay":
            self._update_decay()
        elif self.config.strategy == "exp_decay":
            self._update_exp_decay()

    def _update_progressive(self) -> None:
        """Progressive sliding window update."""
        self.cur_iter_in_group += 1

        if self.cur_iter_in_group >= self.config.iters_per_group:
            self.cur_iter_in_group = 0

            # Advance window
            if self.config.overlap:
                self.cur_timestep += self.config.overlap_step
            else:
                self.cur_timestep += self.config.group_size

        # Handle boundary
        if self.cur_timestep > self.num_timesteps - self.config.group_size:
            if self.config.roll_back:
                self._roll_back()
            else:
                self.cur_timestep = max(0, self.num_timesteps - self.config.group_size)

    def _update_random(self, step: int) -> None:
        """Random timestep selection."""
        rng = np.random.default_rng(step)
        max_start = max(0, self.num_timesteps - self.config.group_size)
        self.cur_timestep = rng.integers(0, max_start + 1)

    def _update_decay(self) -> None:
        """Linear decay of iterations per group."""
        self.cur_iter_in_group += 1

        # Dynamic iterations based on progress
        current_iters = self._get_decay_iters()

        if self.cur_iter_in_group >= current_iters:
            self.cur_iter_in_group = 0

            if self.config.overlap:
                self.cur_timestep += self.config.overlap_step
            else:
                self.cur_timestep += self.config.group_size

        # Handle boundary
        if self.cur_timestep > self.num_timesteps - self.config.group_size:
            if self.config.roll_back:
                self._roll_back()
            else:
                self.cur_timestep = max(0, self.num_timesteps - self.config.group_size)

    def _update_exp_decay(self) -> None:
        """Exponential decay of iterations per group."""
        self.cur_iter_in_group += 1

        # Dynamic iterations based on exponential decay
        current_iters = self._get_exp_decay_iters()

        if self.cur_iter_in_group >= current_iters:
            self.cur_iter_in_group = 0

            if self.config.overlap:
                self.cur_timestep += self.config.overlap_step
            else:
                self.cur_timestep += self.config.group_size

        # Handle boundary
        if self.cur_timestep > self.num_timesteps - self.config.group_size:
            if self.config.roll_back:
                self._roll_back()
            else:
                self.cur_timestep = max(0, self.num_timesteps - self.config.group_size)

    def _get_decay_iters(self) -> int:
        """Calculate iterations for linear decay strategy."""
        if self.config.strategy != "decay":
            return self.config.iters_per_group

        # Linear interpolation based on progress
        progress = self.cur_timestep / max(1, self.num_timesteps - self.config.group_size)
        current_iters = int(
            self.config.max_iters_per_group * (1 - progress)
            + self.config.min_iters_per_group * progress
        )
        return max(self.config.min_iters_per_group, current_iters)

    def _get_exp_decay_iters(self) -> int:
        """Calculate iterations for exponential decay strategy."""
        if self.config.strategy != "exp_decay":
            return self.config.iters_per_group

        # Exponential decay: y(t) = iters * exp(-k * ReLU(t - threshold))
        relu_value = max(0, self.cur_timestep - self.config.exp_decay_threshold)
        decay_value = self.config.iters_per_group * np.exp(
            -self.config.exp_decay_k * relu_value
        )
        return int(np.ceil(decay_value))

    def _roll_back(self) -> None:
        """Roll back to initial timestep."""
        self.cur_timestep = self.init_timestep
        self.cur_iter_in_group = 0

    def is_training_complete(self) -> bool:
        """Check if window has traversed all timesteps."""
        if self.config.strategy in ("progressive", "decay", "exp_decay"):
            return self.cur_timestep >= self.num_timesteps - self.config.group_size
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            "num_timesteps": self.num_timesteps,
            "cur_timestep": self.cur_timestep,
            "cur_iter_in_group": self.cur_iter_in_group,
            "init_timestep": self.init_timestep,
            "config": {
                "strategy": self.config.strategy,
                "group_size": self.config.group_size,
                "iters_per_group": self.config.iters_per_group,
                "overlap": self.config.overlap,
                "overlap_step": self.config.overlap_step,
                "roll_back": self.config.roll_back,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        super().load_state_dict(state_dict)
        self.cur_timestep = state_dict.get("cur_timestep", 0)
        self.cur_iter_in_group = state_dict.get("cur_iter_in_group", 0)
        self.init_timestep = state_dict.get("init_timestep", 0)


# Registry for parameter-driven selection
SCHEDULER_REGISTRY: Dict[str, Type[TimestepScheduler]] = {
    "all": AllSDEScheduler,
    "window": WindowScheduler,
}


def get_scheduler(
    scheduler_type: str,
    num_timesteps: int,
    timestep_fraction: float = 1.0,
    **kwargs: Any,
) -> TimestepScheduler:
    """
    Factory function for creating timestep schedulers.

    Args:
        scheduler_type: Type of scheduler ("all", "window")
        num_timesteps: Total number of denoising timesteps
        timestep_fraction: Fraction of timesteps to train (DanceGRPO: 0.6)
        **kwargs: Additional arguments for scheduler/config

    Returns:
        TimestepScheduler instance

    Example:
        # All SDE (standard GRPO)
        scheduler = get_scheduler("all", num_timesteps=50)

        # All SDE with timestep_fraction (DanceGRPO)
        scheduler = get_scheduler("all", num_timesteps=50, timestep_fraction=0.6)

        # Window scheduler (MixGRPO)
        scheduler = get_scheduler(
            "window",
            num_timesteps=50,
            strategy="progressive",
            group_size=4,
            iters_per_group=25,
        )
    """
    if scheduler_type == "all":
        return AllSDEScheduler(num_timesteps, timestep_fraction=timestep_fraction)
    elif scheduler_type == "window":
        # Build config from kwargs
        config = WindowConfig(
            strategy=kwargs.get("strategy", "progressive"),
            group_size=kwargs.get("group_size", 4),
            iters_per_group=kwargs.get("iters_per_group", 25),
            overlap=kwargs.get("overlap", False),
            overlap_step=kwargs.get("overlap_step", 1),
            roll_back=kwargs.get("roll_back", False),
            max_iters_per_group=kwargs.get("max_iters_per_group"),
            min_iters_per_group=kwargs.get("min_iters_per_group"),
            exp_decay_threshold=kwargs.get("exp_decay_threshold", 13),
            exp_decay_k=kwargs.get("exp_decay_k", 0.1),
        )
        return WindowScheduler(num_timesteps, config)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Available: {list(SCHEDULER_REGISTRY.keys())}")
