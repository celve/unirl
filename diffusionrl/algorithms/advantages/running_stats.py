"""
Running statistics tracker for cross-batch advantage computation.

This module provides a RunningMeanStd class that accumulates mean and std
statistics across multiple batches using Welford's online algorithm.

Used by DanceGRPO and other algorithms that need global_std normalization
with cross-batch statistics.

Reference:
- DanceGRPO: Uses running_reward_stats for stable advantage computation
- Welford's algorithm for numerically stable online variance computation
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


class RunningMeanStd:
    """
    Running mean and standard deviation tracker using Welford's online algorithm.

    Maintains numerically stable running statistics across multiple batches,
    enabling consistent global normalization even when rewards drift over time.

    Example:
        stats = RunningMeanStd()

        # In training loop:
        for batch in batches:
            rewards = compute_rewards(batch)
            stats.update(rewards)

            # Use running stats for normalization
            mean, std = stats.mean, stats.std
            advantages = (rewards - mean) / (std + eps)

    Attributes:
        mean: Current running mean
        var: Current running variance
        std: Current running standard deviation
        count: Total number of samples seen
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        shape: Tuple[int, ...] = (),
    ):
        """
        Initialize running statistics tracker.

        Args:
            epsilon: Small value for numerical stability in std computation
            shape: Shape for element-wise statistics (empty tuple for scalar)
        """
        self.epsilon = epsilon
        self.shape = shape

        # Initialize statistics
        self._mean = np.zeros(shape, dtype=np.float64)
        self._var = np.ones(shape, dtype=np.float64)
        self._count = 0

    @property
    def mean(self) -> float:
        """Get current running mean (scalar)."""
        return float(np.mean(self._mean))

    @property
    def var(self) -> float:
        """Get current running variance (scalar)."""
        return float(np.mean(self._var))

    @property
    def std(self) -> float:
        """Get current running standard deviation with epsilon for stability."""
        return float(np.sqrt(self.var) + self.epsilon)

    @property
    def count(self) -> int:
        """Get total number of samples seen."""
        return self._count

    def update(self, values: Union[np.ndarray, torch.Tensor, float]) -> None:
        """
        Update running statistics with new values using Welford's algorithm.

        Welford's algorithm provides numerically stable online computation:
        - New mean = old mean + (x - old mean) / n
        - New variance uses delta formulation for stability

        Args:
            values: New values to incorporate (numpy array, tensor, or scalar)
        """
        # Convert to numpy array
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, (int, float)):
            values = np.array([values])

        values = values.astype(np.float64).flatten()
        batch_count = len(values)

        if batch_count == 0:
            return

        batch_mean = np.mean(values)
        batch_var = np.var(values) if batch_count > 1 else 0.0

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: float,
        batch_var: float,
        batch_count: int,
    ) -> None:
        """
        Update running statistics from batch moments (parallel algorithm).

        Uses the parallel variance algorithm to combine running stats with batch stats:
        - Combined mean = weighted average
        - Combined variance uses delta^2 correction term

        Args:
            batch_mean: Mean of new batch
            batch_var: Variance of new batch
            batch_count: Size of new batch
        """
        if self._count == 0:
            # First batch: just use batch statistics
            self._mean = np.array([batch_mean], dtype=np.float64)
            self._var = np.array([batch_var], dtype=np.float64)
            self._count = batch_count
            return

        # Parallel variance algorithm
        delta = batch_mean - self._mean
        total_count = self._count + batch_count

        # New mean (weighted average)
        new_mean = self._mean + delta * batch_count / total_count

        # New variance (parallel algorithm with delta correction)
        # M2 = var * count (sum of squared differences)
        m2_a = self._var * self._count
        m2_b = batch_var * batch_count
        # Combined M2 includes cross-term delta^2 correction
        m2_combined = m2_a + m2_b + (delta ** 2) * self._count * batch_count / total_count
        new_var = m2_combined / total_count

        self._mean = new_mean
        self._var = new_var
        self._count = total_count

    def normalize(
        self,
        values: Union[np.ndarray, torch.Tensor],
        update: bool = True,
    ) -> torch.Tensor:
        """
        Normalize values using running statistics.

        Optionally updates statistics with the input values before normalizing.

        Args:
            values: Values to normalize
            update: If True, update running stats with these values first

        Returns:
            Normalized tensor
        """
        if update:
            self.update(values)

        # Handle torch tensor
        if isinstance(values, torch.Tensor):
            device = values.device
            dtype = values.dtype
            normalized = (values - self.mean) / self.std
            return normalized

        # Handle numpy array
        normalized = (values - self.mean) / self.std
        return torch.tensor(normalized, dtype=torch.float32)

    def reset(self) -> None:
        """Reset all statistics."""
        self._mean = np.zeros(self.shape, dtype=np.float64)
        self._var = np.ones(self.shape, dtype=np.float64)
        self._count = 0

    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            "mean": self._mean.copy(),
            "var": self._var.copy(),
            "count": self._count,
            "epsilon": self.epsilon,
            "shape": self.shape,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint."""
        self._mean = np.array(state_dict.get("mean", self._mean), dtype=np.float64)
        self._var = np.array(state_dict.get("var", self._var), dtype=np.float64)
        self._count = state_dict.get("count", 0)
        self.epsilon = state_dict.get("epsilon", self.epsilon)

    def __repr__(self) -> str:
        return f"RunningMeanStd(mean={self.mean:.4f}, std={self.std:.4f}, count={self._count})"


class RunningRewardNormalizer:
    """
    High-level reward normalizer with running statistics.

    Provides a clean interface for algorithms that need cross-batch
    reward normalization (e.g., DanceGRPO's global_std mode).

    Example:
        normalizer = RunningRewardNormalizer(clip_max=5.0)

        # In training loop:
        advantages = normalizer.normalize(rewards, update_stats=True)
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        warmup_steps: int = 0,
    ):
        """
        Initialize reward normalizer.

        Args:
            epsilon: Small value for numerical stability
            clip_max: If set, clip advantages to [-clip_max, clip_max]
            warmup_steps: Number of batches before using running stats
        """
        self.running_stats = RunningMeanStd(epsilon=epsilon)
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.warmup_steps = warmup_steps
        self._step_count = 0

    def normalize(
        self,
        rewards: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """
        Normalize rewards using running statistics.

        During warmup, uses batch statistics only. After warmup,
        uses accumulated running statistics for stable normalization.

        Args:
            rewards: Reward tensor [B]
            update_stats: If True, update running stats with these rewards

        Returns:
            Normalized advantage tensor [B]
        """
        device = rewards.device
        dtype = rewards.dtype

        if update_stats:
            self.running_stats.update(rewards)
            self._step_count += 1

        # During warmup, use batch-only statistics
        if self._step_count <= self.warmup_steps:
            mean = rewards.mean()
            std = rewards.std() + self.epsilon
        else:
            # Use running statistics
            mean = self.running_stats.mean
            std = self.running_stats.std

        advantages = (rewards - mean) / std

        if self.clip_max is not None:
            advantages = advantages.clamp(-self.clip_max, self.clip_max)

        return advantages

    def reset(self) -> None:
        """Reset all statistics and step counter."""
        self.running_stats.reset()
        self._step_count = 0

    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            "running_stats": self.running_stats.state_dict(),
            "step_count": self._step_count,
            "clip_max": self.clip_max,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint."""
        if "running_stats" in state_dict:
            self.running_stats.load_state_dict(state_dict["running_stats"])
        self._step_count = state_dict.get("step_count", 0)
        self.clip_max = state_dict.get("clip_max", self.clip_max)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
