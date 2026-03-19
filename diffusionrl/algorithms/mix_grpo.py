"""
MixGRPO Algorithm Implementation.

Mixed SDE/ODE GRPO with configurable ratio.
"""
from typing import Any, Dict, Optional, Set

from diffusionrl.config.build_domain_args import resolve_sde_schedule_config
from .base import SamplingRequirements
from .grpo import GRPOAlgorithm, _resolve_algorithm_sde_config


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
                When enabled, resolve_training_indices returns SDE indices.
                When disabled, trains on all timesteps (standard behavior).
            **kwargs: Additional GRPO arguments
        """
        super().__init__(**kwargs)
        self.sde_ratio = sde_ratio
        self.window_training = window_training

    @classmethod
    def from_config(cls, config: dict) -> "MixGRPOAlgorithm":
        extra = dict(config.get("algorithm_kwargs") or {})
        sde_config = _resolve_algorithm_sde_config(config)
        sde_schedule_config = resolve_sde_schedule_config(config)
        known_keys = {
            "clip_range",
            "clip_schedule",
            "use_kl_penalty",
            "kl_coef",
            "component_mix_stage",
            "samples_per_prompt",
            "eval_ema_decay",
            "eval_ema_update_interval",
            "ratio_reg_coef",
            "skip_last_timestep",
            "skip_initial_timesteps",
            "model_type",
            "adv_normalization",
            "adv_norm_eps",
            "adv_clip_abs",
            "use_global_std",
            "trimmed_ratio",
            "window_training",
        }
        runtime_only_keys = {
            "shuffle_samples",
            "shuffle_seed",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys and key not in runtime_only_keys)
        if unknown:
            import warnings

            warnings.warn(
                f"MixGRPOAlgorithm.from_config received unknown algorithm_kwargs keys: {unknown}. "
                "These keys are ignored by MixGRPO algorithm constructor.",
                stacklevel=3,
            )

        return cls(
            clip_range=float(extra.get("clip_range", 1e-4)),
            clip_schedule=str(extra.get("clip_schedule", "constant")),
            use_kl_penalty=bool(extra.get("use_kl_penalty", True)),
            kl_coef=float(extra.get("kl_coef", 0.01)),
            component_mix_stage=str(extra.get("component_mix_stage", "reward")),
            samples_per_prompt=int(extra.get("samples_per_prompt", 1)),
            eval_ema_decay=float(extra.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(extra.get("eval_ema_update_interval", 1)),
            ratio_reg_coef=float(extra.get("ratio_reg_coef", 0.0)),
            sde_config=sde_config,
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            model_type=str(extra.get("model_type", "default")),
            adv_normalization=str(extra.get("adv_normalization", "group")),
            epsilon=float(extra.get("adv_norm_eps", 1e-8)),
            clip_max=extra.get("adv_clip_abs", 5.0),
            use_global_std=bool(extra.get("use_global_std", False)),
            trimmed_ratio=float(extra.get("trimmed_ratio", 0.0)),
            sde_ratio=float(sde_schedule_config.sde_ratio),
            window_training=bool(extra.get("window_training", False)),
        )

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return MixGRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
            extras={"sde_ratio": self.sde_ratio},
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

    def resolve_training_indices(
        self,
        *,
        num_steps: int,
        sde_indices: Optional[Set[int]] = None,
    ) -> Set[int]:
        """Resolve the timestep indices that should contribute to training."""
        if self.window_training:
            if sde_indices is not None:
                return set(int(i) for i in sde_indices)
            return self.get_sde_indices(num_steps)
        return set(range(num_steps))
