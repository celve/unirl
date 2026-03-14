"""
Per-Prompt Statistics Tracker for cross-batch advantage computation.

This module provides a stateful tracker that accumulates reward statistics
across multiple batches for each unique prompt. This enables per-prompt
normalization even when samples for the same prompt appear in different batches.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class PerPromptStatTracker:
    """
    Tracks per-prompt reward statistics across multiple batches.

    This tracker maintains a running buffer of rewards for each unique prompt,
    allowing computation of mean/std that accounts for historical data beyond
    the current batch.

    Example:
        tracker = PerPromptStatTracker(buffer_size=16, min_count=2)

        # In training loop:
        advantages = tracker.compute_advantages(prompt_ids, rewards)
        tracker.update(prompt_ids, rewards)

    Attributes:
        buffer_size: Maximum number of rewards to keep per prompt
        min_count: Minimum samples required before using per-prompt stats
    """

    def __init__(
        self,
        buffer_size: int = 16,
        min_count: int = 2,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_global_std: bool = False,
        fallback_mean: float = 0.0,
        fallback_std: float = 1.0,
    ):
        """
        Initialize per-prompt statistics tracker.

        Args:
            buffer_size: Maximum number of rewards to store per prompt
            min_count: Minimum number of samples before using per-prompt stats
            epsilon: Small value for numerical stability
            clip_max: If set, clip advantages to [-clip_max, clip_max]
            fallback_mean: Mean to use when insufficient samples
            fallback_std: Std to use when insufficient samples
        """
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.use_global_std = use_global_std
        self.fallback_mean = fallback_mean
        self.fallback_std = fallback_std

        # Storage: prompt_id -> list of historical rewards
        self.prompt_stats: Dict[str, List[float]] = defaultdict(list)

        # Running global statistics (for fallback)
        self._global_rewards: List[float] = []
        self._global_buffer_size = buffer_size * 10

    def update(self, prompt_ids: List[str], rewards: torch.Tensor) -> None:
        """
        Update statistics with new rewards.

        Args:
            prompt_ids: List of prompt identifiers
            rewards: Reward tensor [B]
        """
        rewards_list = rewards.detach().cpu().tolist()

        for pid, r in zip(prompt_ids, rewards_list):
            buf = self.prompt_stats[pid]
            buf.append(r)

            # Maintain buffer size
            if len(buf) > self.buffer_size:
                buf.pop(0)

        # Update global statistics
        self._global_rewards.extend(rewards_list)
        if len(self._global_rewards) > self._global_buffer_size:
            self._global_rewards = self._global_rewards[-self._global_buffer_size:]

    def compute_advantages(
        self,
        prompt_ids: List[str],
        rewards: torch.Tensor,
        update_stats: bool = False,
    ) -> torch.Tensor:
        """
        Compute advantages using per-prompt statistics.

        For prompts with sufficient history, uses per-prompt mean/std.
        For new prompts, falls back to current-batch per-prompt group
        normalization so that the first rollout still gets meaningful
        (zero-mean, unit-variance) advantages instead of raw rewards.

        Args:
            prompt_ids: List of prompt identifiers [B]
            rewards: Reward tensor [B]
            update_stats: If True, also update statistics (combines compute and update)

        Returns:
            Advantage tensor [B]
        """
        device = rewards.device
        dtype = rewards.dtype

        # Optionally use global std across current rewards (flow_grpo global_std behavior)
        global_std = None
        if self.use_global_std:
            try:
                global_std = rewards.std().item()
            except Exception:
                global_std = None

        # Pre-compute current-batch per-prompt groups for fallback
        rewards_list = rewards.tolist()
        batch_prompt_rewards: Dict[str, List[float]] = defaultdict(list)
        for pid, r in zip(prompt_ids, rewards_list):
            batch_prompt_rewards[pid].append(r)

        advantages = []

        for pid, r in zip(prompt_ids, rewards_list):
            buf = self.prompt_stats.get(pid, [])

            if len(buf) >= self.min_count:
                # Use per-prompt historical statistics
                mean = sum(buf) / len(buf)
                if self.use_global_std and global_std is not None:
                    std = max(global_std, self.epsilon)
                else:
                    variance = sum((x - mean) ** 2 for x in buf) / len(buf)
                    std = max(variance ** 0.5, self.epsilon)
                adv = (r - mean) / std
            else:
                # Fall back to current-batch per-prompt group normalization
                group = batch_prompt_rewards[pid]
                if len(group) >= 2:
                    mean = sum(group) / len(group)
                    if self.use_global_std and global_std is not None:
                        std = max(global_std, self.epsilon)
                    else:
                        variance = sum((x - mean) ** 2 for x in group) / len(group)
                        std = max(variance ** 0.5, self.epsilon)
                    adv = (r - mean) / std
                else:
                    # Single sample for this prompt in the batch – no
                    # normalization possible; set advantage to zero.
                    adv = 0.0

            advantages.append(adv)

        advantages = torch.tensor(advantages, dtype=dtype, device=device)

        # Clip if configured
        if self.clip_max is not None:
            advantages = advantages.clamp(-self.clip_max, self.clip_max)

        # Optionally update statistics
        if update_stats:
            self.update(prompt_ids, rewards)

        return advantages

    def _get_global_stats(self) -> Tuple[float, float]:
        """Get global mean and std for fallback."""
        if len(self._global_rewards) >= self.min_count:
            mean = sum(self._global_rewards) / len(self._global_rewards)
            variance = sum((x - mean) ** 2 for x in self._global_rewards) / len(self._global_rewards)
            std = max(variance ** 0.5, self.epsilon)
            return mean, std
        return self.fallback_mean, self.fallback_std

    def reset(self) -> None:
        """Reset all statistics."""
        self.prompt_stats.clear()
        self._global_rewards.clear()

    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            "prompt_stats": dict(self.prompt_stats),
            "global_rewards": self._global_rewards.copy(),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint."""
        self.prompt_stats = defaultdict(list, state_dict.get("prompt_stats", {}))
        self._global_rewards = state_dict.get("global_rewards", [])
