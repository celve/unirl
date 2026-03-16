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
    def from_config(cls, config: dict) -> "MixGRPOAlgorithm":
        extra = dict(config.get("algorithm_kwargs") or {})
        known_keys = {
            "clip_range",
            "clip_schedule",
            "use_kl_penalty",
            "kl_coef",
            "samples_per_prompt",
            "eval_ema_decay",
            "eval_ema_update_interval",
            "ratio_reg_coef",
            "eta",
            "sde_type",
            "time_shift",
            "skip_last_timestep",
            "skip_initial_timesteps",
            "model_type",
            "adv_normalization",
            "adv_norm_eps",
            "adv_clip_abs",
            "use_global_std",
            "trimmed_ratio",
            "sde_ratio",
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
            samples_per_prompt=int(extra.get("samples_per_prompt", 1)),
            eval_ema_decay=float(extra.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(extra.get("eval_ema_update_interval", 1)),
            ratio_reg_coef=float(extra.get("ratio_reg_coef", 0.0)),
            eta=float(extra.get("eta", 0.7)),
            sde_type=str(extra.get("sde_type", "sde")),
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            model_type=str(extra.get("model_type", "default")),
            adv_normalization=str(extra.get("adv_normalization", "group")),
            epsilon=float(extra.get("adv_norm_eps", 1e-8)),
            clip_max=extra.get("adv_clip_abs", 5.0),
            use_global_std=bool(extra.get("use_global_std", False)),
            trimmed_ratio=float(extra.get("trimmed_ratio", 0.0)),
            sde_ratio=float(extra.get("sde_ratio", 1.0)),
            window_training=bool(extra.get("window_training", False)),
        )

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return MixGRPO sampling requirements."""
        return self._build_sampling_requirements(extras={"sde_ratio": self.sde_ratio})

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
