"""
NFT (Negative Fine-Tuning) Algorithm Implementation.

DiffusionNFT forward process diffusion RL.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseAlgorithm, SamplingRequirements


class NFTAlgorithm(BaseAlgorithm):
    """
    NFT (Negative Fine-Tuning) Algorithm - DiffusionNFT.

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
            epsilon=epsilon,
            clip_max=clip_max,
            use_per_prompt_tracker=use_per_prompt_tracker,
            per_prompt_buffer_size=per_prompt_buffer_size,
            per_prompt_min_count=per_prompt_min_count,
            use_global_std=use_global_std,
            **kwargs,
        )
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.adv_mode = adv_mode
        self.use_adaptive_weight = use_adaptive_weight
        self.shift = shift
        self.ema_decay = ema_decay

        # Loss function (lazy load)
        self._loss_fn = None

    @classmethod
    def from_args(cls, args: Any) -> "NFTAlgorithm":
        """Construct NFT algorithm from runtime args."""
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "beta": 0.1,
                "adv_clip_max": 5.0,
                "adv_mode": "raw",
                "use_adaptive_weight": True,
                "shift": getattr(args, "shift", 3.0),
                "ema_decay": 0.001,
                "use_per_prompt_tracker": getattr(args, "use_per_prompt_stat_tracker", False),
                "per_prompt_buffer_size": getattr(args, "per_prompt_buffer_size", 16),
                "per_prompt_min_count": getattr(args, "per_prompt_min_count", 2),
                "use_global_std": getattr(args, "use_global_std", False),
            }
        )
        kwargs.update(cls._algorithm_kwargs_from_args(args))
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
            requires_trajectory=False,
            requires_log_prob=False,
            extras={
                "sde_ratio": 0.0,
                "requires_clean_latents": True,
                "forward_diffusion_in_loss": True,
            },
        )

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
