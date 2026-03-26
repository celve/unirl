"""
Timestep schedulers for MixGRPO training.

This module implements sliding window timestep scheduling, which determines
which timesteps use SDE (stochastic) vs ODE (deterministic) sampling.

Key concepts:
- In MixGRPO, only a subset of timesteps use SDE sampling
- SDE timesteps compute log_prob and contribute to policy gradient loss
- ODE timesteps are faster but don't contribute to training

Based on the original MixGRPO timestep-window scheduling logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Dict, Any, Tuple, Type, Union

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

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {"num_timesteps": self.num_timesteps}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.num_timesteps = state_dict.get("num_timesteps", self.num_timesteps)


def _normalize_timestep_fraction(
    timestep_fraction: Union[float, Tuple[float, float], List[float]],
) -> Tuple[float, float]:
    """Normalize timestep_fraction to a (start, end) tuple.

    Supports:
    - Single float x  -> (0.0, x)
    - Tuple/list (x, y) -> (x, y)

    Returns:
        (start_fraction, end_fraction) both in [0.0, 1.0], start <= end.
    """
    if isinstance(timestep_fraction, (list, tuple)):
        if len(timestep_fraction) != 2:
            raise ValueError(
                f"timestep_fraction tuple must have exactly 2 elements, got {len(timestep_fraction)}"
            )
        start, end = float(timestep_fraction[0]), float(timestep_fraction[1])
    else:
        start, end = 0.0, float(timestep_fraction)
    if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
        raise ValueError(
            f"timestep_fraction values must be in [0.0, 1.0], got ({start}, {end})"
        )
    if start > end:
        raise ValueError(
            f"timestep_fraction start ({start}) must be <= end ({end})"
        )
    return (start, end)


class AllSDEScheduler(TimestepScheduler):
    """
    All SDE scheduler (standard GRPO) with optional timestep_fraction (DanceGRPO)
    and optional num_sde_steps (random sparse SDE).

    When timestep_fraction=1.0 (default): All timesteps use SDE sampling.
    When timestep_fraction<1.0 (single float): Only the first fraction of timesteps use SDE.
    When timestep_fraction=(x, y) (tuple): Only timesteps in [x, y) fraction range use SDE.

    When num_sde_steps is set (int): Instead of using ALL timesteps in the fraction
    range, randomly sample num_sde_steps non-contiguous timesteps from the range each
    rollout. The seed is derived from the rollout step so each rollout gets a fresh
    random selection.  Remaining timesteps in the range fall back to ODE.
    """

    def __init__(
        self,
        num_timesteps: int,
        timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
        num_sde_steps: Optional[int] = None,
    ):
        """
        Initialize AllSDE scheduler.

        Args:
            num_timesteps: Total number of denoising timesteps
            timestep_fraction: Fraction of timesteps to train.
                Single float x means [0, x) range (backward compatible).
                Tuple (x, y) means [x, y) range.
            num_sde_steps: If set, randomly pick this many SDE steps from the
                fraction range each rollout.  Must be <= number of steps in range.
        """
        super().__init__(num_timesteps)
        self.timestep_fraction = timestep_fraction
        self.num_sde_steps = num_sde_steps
        self._fraction_start, self._fraction_end = _normalize_timestep_fraction(timestep_fraction)
        self._effective_start = int(num_timesteps * self._fraction_start)
        self._effective_end = int(num_timesteps * self._fraction_end)
        # Validate num_sde_steps
        if num_sde_steps is not None:
            pool_size = self._effective_end - self._effective_start
            if num_sde_steps > pool_size:
                raise ValueError(
                    f"num_sde_steps ({num_sde_steps}) exceeds available timesteps "
                    f"in fraction range [{self._effective_start}, {self._effective_end}) "
                    f"(pool_size={pool_size})"
                )
            if num_sde_steps <= 0:
                raise ValueError(f"num_sde_steps must be positive, got {num_sde_steps}")

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Return timestep indices in [start, end) fraction range.

        If num_sde_steps is set, randomly sub-sample from the range using
        the rollout step as seed for reproducibility.
        """
        pool = list(range(self._effective_start, self._effective_end))
        if self.num_sde_steps is None or self.num_sde_steps >= len(pool):
            return set(pool)
        # Use rollout step as seed for per-rollout randomness
        seed = step if step is not None else 0
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pool, size=self.num_sde_steps, replace=False)
        return set(int(i) for i in chosen)

    def update(self, step: int) -> None:
        """No-op for all SDE scheduler."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            "num_timesteps": self.num_timesteps,
            "timestep_fraction": self.timestep_fraction,
            "num_sde_steps": self.num_sde_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        super().load_state_dict(state_dict)
        raw = state_dict.get("timestep_fraction", 1.0)
        self.timestep_fraction = raw
        self.num_sde_steps = state_dict.get("num_sde_steps", None)
        self._fraction_start, self._fraction_end = _normalize_timestep_fraction(raw)
        self._effective_start = int(self.num_timesteps * self._fraction_start)
        self._effective_end = int(self.num_timesteps * self._fraction_end)


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
    timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
    num_sde_steps: Optional[int] = None,
    **kwargs: Any,
) -> TimestepScheduler:
    """
    Factory function for creating timestep schedulers.

    Args:
        scheduler_type: Type of scheduler ("all", "window")
        num_timesteps: Total number of denoising timesteps
        timestep_fraction: Fraction of timesteps to train.
            Single float x: SDE on [0, x) range (e.g. 0.6 = first 60%).
            Tuple (x, y): SDE on [x, y) range (e.g. (0.2, 0.8) = 20%-80%).
        num_sde_steps: If set, randomly pick this many SDE steps from the
            fraction range each rollout (only for "all" scheduler).
        **kwargs: Additional arguments for scheduler/config

    Returns:
        TimestepScheduler instance

    Example:
        # All SDE (standard GRPO)
        scheduler = get_scheduler("all", num_timesteps=50)

        # All SDE with timestep_fraction (DanceGRPO)
        scheduler = get_scheduler("all", num_timesteps=50, timestep_fraction=0.6)

        # Random sparse SDE: pick 3 random steps from [0.1, 0.3) each rollout
        scheduler = get_scheduler("all", num_timesteps=50, timestep_fraction=(0.1, 0.3), num_sde_steps=3)

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
        return AllSDEScheduler(
            num_timesteps,
            timestep_fraction=timestep_fraction,
            num_sde_steps=num_sde_steps,
        )
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
