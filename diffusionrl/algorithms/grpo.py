"""
GRPO Algorithm Implementation.

Standard GRPO with group normalization for advantages.
"""
from typing import Any, Dict, List, Optional

import torch

from .base import BaseAlgorithm, SamplingRequirements


class GRPOAlgorithm(BaseAlgorithm):
    """
    Standard GRPO Algorithm.

    Features:
    - Group normalization for advantages (within prompt groups)
    - PPO-style clipped objective
    - Optional KL penalty

    Reference: DanceGRPO
    """

    def __init__(
        self,
        clip_range: float = 1e-4,
        kl_coef: float = 0.01,
        advantage_type: str = "group",
        eta: float = 1.0,
        sde_type: str = "sde",
        epsilon: float = 1e-4,
        clip_max: float = 5.0,
        use_per_prompt_tracker: bool = False,
        per_prompt_mode: str = "running",
        per_prompt_buffer_size: int = 16,
        per_prompt_min_count: int = 2,
        use_running_stats: bool = False,
        running_stats_warmup: int = 0,
        use_global_std: bool = False,
        ignore_last: bool = False,
        frozen_init_timesteps: int = 0,
        **kwargs,
    ):
        """
        Initialize GRPO algorithm.

        Args:
            clip_range: PPO clip range
            kl_coef: KL penalty coefficient
            advantage_type: Advantage normalization type ("global", "group", "per_prompt")
            eta: SDE noise coefficient
            sde_type: Type of SDE ("sde", "cps", "dance")
            epsilon: Small value for numerical stability
            clip_max: Maximum advantage clip value (optional)
            use_per_prompt_tracker: Use PerPromptStatTracker for cross-batch stats
            per_prompt_mode: "running" (tracker) or "batch" (per-batch stats)
            per_prompt_buffer_size: Buffer size for per-prompt tracker
            per_prompt_min_count: Min samples before using per-prompt stats
            use_running_stats: Use RunningMeanStd for cross-batch global normalization (DanceGRPO)
            running_stats_warmup: Warmup batches before using running stats
            ignore_last: Skip the last timestep (t->0) in training (MixGRPO).
                The last step has very low noise level, causing unstable log_prob.
            frozen_init_timesteps: Skip the first N timesteps in training (MixGRPO).
                Early timesteps may have high variance.
            **kwargs: Additional arguments
        """
        super().__init__(
            clip_range=clip_range,
            kl_coef=kl_coef,
            advantage_type=advantage_type,
            epsilon=epsilon,
            clip_max=clip_max,
            use_per_prompt_tracker=use_per_prompt_tracker,
            per_prompt_mode=per_prompt_mode,
            per_prompt_buffer_size=per_prompt_buffer_size,
            per_prompt_min_count=per_prompt_min_count,
            use_running_stats=use_running_stats,
            running_stats_warmup=running_stats_warmup,
            use_global_std=use_global_std,
            **kwargs,
        )
        self.eta = eta
        self.sde_type = sde_type

        # MixGRPO stability controls
        self.ignore_last = ignore_last
        self.frozen_init_timesteps = frozen_init_timesteps

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using group normalization.

        Extends the base implementation to support GRPO-specific
        per_prompt_mode="batch" normalization.

        Args:
            rewards: Reward tensor [batch_size]
            num_samples_per_prompt: Number of samples per prompt
            prompts: Optional list of prompt strings (for per_prompt with tracker)

        Returns:
            Normalized advantage tensor [batch_size]
        """
        if self.advantage_type == "per_prompt" and self.per_prompt_mode == "batch":
            return self._normalize_per_prompt_batch(
                rewards, num_samples_per_prompt, prompts
            )
        return super().compute_advantages(rewards, num_samples_per_prompt, prompts)

    def _normalize_per_prompt_batch(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Per-prompt normalization using current batch statistics.

        - Mean is computed per prompt group.
        - Std is computed per prompt group unless use_global_std=True.
        """
        if prompts is None:
            return self._normalize_group(rewards, num_samples_per_prompt)

        # Expand prompts to per-sample list if needed
        if len(prompts) * num_samples_per_prompt == len(rewards):
            prompts = [p for p in prompts for _ in range(num_samples_per_prompt)]
        elif len(prompts) != len(rewards):
            return self._normalize_group(rewards, num_samples_per_prompt)

        # Build prompt -> indices mapping
        prompt_to_indices: Dict[str, List[int]] = {}
        for idx, prompt in enumerate(prompts):
            prompt_to_indices.setdefault(prompt, []).append(idx)

        # Global std (batch or running stats)
        global_std = None
        if self.use_global_std:
            global_std = self._get_global_std(rewards)

        advantages = torch.empty_like(rewards)
        for prompt, indices in prompt_to_indices.items():
            idx_tensor = torch.tensor(indices, device=rewards.device, dtype=torch.long)
            prompt_rewards = rewards.index_select(0, idx_tensor)
            mean = prompt_rewards.mean()
            if global_std is None:
                std = prompt_rewards.std() + self.epsilon
            else:
                std = global_std
            prompt_adv = (prompt_rewards - mean) / std
            advantages.index_copy_(0, idx_tensor, prompt_adv)

        if self.clip_max is not None:
            advantages = advantages.clamp(-self.clip_max, self.clip_max)

        return advantages

    def _get_global_std(self, rewards: torch.Tensor) -> torch.Tensor:
        """Get global std with optional running stats (DanceGRPO style)."""
        if self.running_reward_normalizer is None:
            return rewards.std() + self.epsilon

        # Update running stats and respect warmup
        self.running_reward_normalizer.running_stats.update(rewards)
        self.running_reward_normalizer._step_count += 1
        if self.running_reward_normalizer._step_count <= self.running_reward_normalizer.warmup_steps:
            return rewards.std() + self.epsilon

        std = self.running_reward_normalizer.running_stats.std
        return torch.tensor(std, device=rewards.device, dtype=rewards.dtype)

    @classmethod
    def _grpo_kwargs_from_args(cls, args: Any) -> Dict[str, Any]:
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "eta": getattr(args.sampling, "eta", 1.0),
                "sde_type": getattr(args.sampling, "sde_type", "sde"),
                "use_per_prompt_tracker": getattr(args.algorithm, "use_per_prompt_stat_tracker", False),
                "per_prompt_mode": getattr(args.algorithm, "per_prompt_mode", "running"),
                "per_prompt_buffer_size": getattr(args.algorithm, "per_prompt_buffer_size", 16),
                "per_prompt_min_count": getattr(args.algorithm, "per_prompt_min_count", 2),
                "use_running_stats": getattr(args.algorithm, "use_running_stats", False),
                "running_stats_warmup": getattr(args.algorithm, "running_stats_warmup", 0),
                "use_global_std": getattr(args.algorithm, "use_global_std", False),
                "trimmed_ratio": getattr(args.algorithm, "trimmed_ratio", 0.0),
                "ignore_last": getattr(args.algorithm, "ignore_last", False),
                "frozen_init_timesteps": getattr(args.algorithm, "frozen_init_timesteps", 0),
            }
        )
        return kwargs

    @classmethod
    def from_args(cls, args: Any) -> "GRPOAlgorithm":
        """Construct GRPO algorithm from runtime args."""
        kwargs = cls._grpo_kwargs_from_args(args)
        kwargs.update(cls._algorithm_kwargs_from_args(args))
        return cls(**kwargs)

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return GRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
        )
