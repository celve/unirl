"""
NFT (Noise-Free Training) Algorithm Implementation.

DiffusionNFT forward process diffusion RL.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseAlgorithm, SamplingRequirements
from diffusionrl.advantages.normalizers import normalize_global, normalize_grouped, build_fixed_size_groups


class NFTAlgorithm(BaseAlgorithm):
    """
    NFT (Noise-Free Training) Algorithm - DiffusionNFT.

    Forward process diffusion RL that optimizes directly on the forward
    diffusion process instead of reverse sampling trajectories.

    Key differences from GRPO:
    - No trajectory storage needed (only clean latents)
    - No log probabilities needed (no importance sampling)
    - Uses dual adapter mechanism (new/old) with EMA update
    - Forward diffusion happens in loss computation

    Args:
        beta: Interpolation weight for positive/negative predictions
        adv_clip_max: Maximum advantage clipping value
        adv_mode: Advantage processing mode ("raw", "sign", "binary", "one_only")
        use_adaptive_weight: Whether to use adaptive weighting
        shift: Time shift parameter for sigma schedule
        ema_decay: EMA decay rate for old adapter update
        kl_coef: KL regularization coefficient
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        beta: float = 0.1,
        adv_clip_max: float = 5.0,
        adv_mode: str = "raw",
        use_adaptive_weight: bool = True,
        shift: float = 3.0,
        ema_decay: float = 0.001,
        kl_coef: float = 0.0,
        advantage_type: str = "group",
        epsilon: float = 1e-4,
        clip_max: float = 5.0,
        use_per_prompt_tracker: bool = False,
        per_prompt_buffer_size: int = 16,
        per_prompt_min_count: int = 2,
        use_global_std: bool = False,
        **kwargs,
    ):
        # Remove clip_range from kwargs if present, NFT doesn't use it
        kwargs.pop('clip_range', None)
        super().__init__(
            clip_range=0.0,  # Not used by NFT
            kl_coef=kl_coef,
            advantage_type=advantage_type,
            **kwargs,
        )
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.adv_mode = adv_mode
        self.use_adaptive_weight = use_adaptive_weight
        self.shift = shift
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.clip_max = clip_max

        # Per-prompt statistics tracker (for per_prompt advantage type)
        self.per_prompt_tracker = None
        if use_per_prompt_tracker or advantage_type == "per_prompt":
            from diffusionrl.advantages.per_prompt_tracker import PerPromptStatTracker
            self.per_prompt_tracker = PerPromptStatTracker(
                buffer_size=per_prompt_buffer_size,
                min_count=per_prompt_min_count,
                epsilon=epsilon,
                clip_max=clip_max,
            )

        # Loss function (lazy load)
        self._loss_fn = None

    @classmethod
    def from_args(cls, args: Any) -> "NFTAlgorithm":
        """Construct NFT algorithm from runtime args."""
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "beta": getattr(args, "nft_beta", 0.1),
                "adv_clip_max": getattr(args, "nft_adv_clip_max", 5.0),
                "adv_mode": getattr(args, "nft_adv_mode", "raw"),
                "use_adaptive_weight": getattr(args, "nft_use_adaptive_weight", True),
                "shift": getattr(args, "shift", 3.0),
                "ema_decay": getattr(args, "ema_decay", 0.001),
                "use_per_prompt_tracker": getattr(args, "use_per_prompt_stat_tracker", False),
                "per_prompt_buffer_size": getattr(args, "per_prompt_buffer_size", 16),
                "per_prompt_min_count": getattr(args, "per_prompt_min_count", 2),
                "use_global_std": getattr(args, "use_global_std", False),
            }
        )
        return cls(**kwargs)

    @property
    def loss_fn(self):
        """Lazy load NFT loss function."""
        if self._loss_fn is None:
            from diffusionrl.losses import NFTLoss
            self._loss_fn = NFTLoss(
                beta=self.beta,
                adv_clip_max=self.adv_clip_max,
                adv_mode=self.adv_mode,
                use_adaptive_weight=self.use_adaptive_weight,
                shift=self.shift,
                kl_coef=self.kl_coef,
            )
        return self._loss_fn

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return NFT sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=False,  # Only needs clean latents
            requires_log_prob=False,  # No importance sampling
            sde_ratio=0.0,  # All ODE for inference (doesn't affect training)
            requires_clean_latents=True,  # NFT needs clean x0
            forward_diffusion_in_loss=True,  # Forward process in loss
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using group normalization.

        Args:
            rewards: Reward tensor [batch_size]
            num_samples_per_prompt: Number of samples per prompt
            prompts: Optional list of prompt strings (for per_prompt with tracker)

        Returns:
            Normalized advantage tensor [batch_size]
        """
        if self.advantage_type == "global":
            return self._normalize_global(rewards)
        elif self.advantage_type == "group":
            return self._normalize_group(rewards, num_samples_per_prompt)
        elif self.advantage_type == "per_prompt":
            # Use per-prompt tracker if available and prompts provided
            if self.per_prompt_tracker is not None and prompts is not None:
                # Expand prompts to match rewards (each prompt repeated num_samples_per_prompt times)
                if len(prompts) * num_samples_per_prompt == len(rewards):
                    expanded_prompts = []
                    for p in prompts:
                        expanded_prompts.extend([p] * num_samples_per_prompt)
                    prompts = expanded_prompts
                return self.per_prompt_tracker.compute_advantages(
                    prompts, rewards, update_stats=True
                )
            # Fall back to batch-level group normalization
            return self._normalize_group(rewards, num_samples_per_prompt)
        else:
            raise ValueError(f"Unknown advantage_type: {self.advantage_type}")

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:
        """Global normalization across all samples."""
        return normalize_global(rewards, epsilon=self.epsilon, clip_max=self.clip_max)

    def _normalize_group(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
    ) -> torch.Tensor:
        """Group normalization within prompt groups."""
        batch_size = rewards.shape[0]
        if num_samples_per_prompt <= 0 or batch_size % num_samples_per_prompt != 0:
            return self._normalize_global(rewards)
        groups = build_fixed_size_groups(batch_size, num_samples_per_prompt)
        return normalize_grouped(rewards, groups, epsilon=self.epsilon, clip_max=self.clip_max)

    def update_old_adapter(self, model: nn.Module) -> bool:
        """
        Update old adapter using EMA from new adapter.

        Should be called after each optimizer step.

        Args:
            model: Model with dual LoRA adapters

        Returns:
            True if update was successful
        """
        return self.loss_fn.update_old_adapter(model, self.ema_decay)

    # ========== NFT-specific hooks ==========

    def requires_ema_update(self) -> bool:
        """NFT requires EMA updates for dual adapter mechanism.

        Returns:
            True - NFT always uses EMA updates
        """
        return True

    def get_ema_decay(self) -> float:
        """Get EMA decay rate.

        Returns:
            The configured ema_decay value
        """
        return self.ema_decay

    def post_optimizer_step_hook(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update old adapter via EMA after optimizer step.

        This implements the dual adapter mechanism where the 'old' adapter
        is updated via EMA from the 'new' adapter after each training step.

        Args:
            model: The model with dual LoRA adapters
            optimizer: The optimizer (unused)
            batch: The training batch (unused)

        Returns:
            Dictionary with EMA update status
        """
        success = self.update_old_adapter(model)
        return {"ema_updated": success}

    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration as dictionary."""
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "adv_clip_max": self.adv_clip_max,
            "adv_mode": self.adv_mode,
            "use_adaptive_weight": self.use_adaptive_weight,
            "shift": self.shift,
            "ema_decay": self.ema_decay,
        })
        return config
