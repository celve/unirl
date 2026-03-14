"""
GRPO Algorithm Implementation.

Standard GRPO with group normalization for advantages.
"""
from typing import Any, Dict

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
        per_prompt_mode: str = "batch",
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

    @classmethod
    def _grpo_kwargs_from_args(cls, args: Any) -> Dict[str, Any]:
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "eta": getattr(args.sampling, "eta", 1.0),
                "sde_type": getattr(args.sampling, "sde_type", "sde"),
                "use_per_prompt_tracker": getattr(args.algorithm, "use_per_prompt_stat_tracker", False),
                "per_prompt_mode": getattr(args.algorithm, "per_prompt_mode", "batch"),
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
