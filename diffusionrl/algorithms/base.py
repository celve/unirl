"""
diffusionrl Algorithm Base Class.

Defines algorithm responsibilities in rollout/advantage pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch

from diffusionrl.models.base import ModelBundle
from diffusionrl.types.sampling import SamplingRequirements

from .normalizers import normalize_global, normalize_grouped

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EMASpec:
    """Algorithm-declared EMA policy.

    The algorithm declares *what* EMA behavior it needs. Runtime owns
    the mechanism that materializes and updates the trackers.
    """

    enable_eval_ema: bool = True
    eval_ema_decay: float = 0.9
    eval_ema_update_interval: int = 1
    reference_mode: str = "none"
    reference_decay: float = 0.0
    reference_decay_type: str = "constant"
    reference_flat_steps: int = 0
    reference_uprate: float = 0.001
    reference_uphold: float = 0.5
    reference_update_timing: str = "optimizer_step"
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"


@dataclass(frozen=True)
class BaseAlgorithmConfig:
    """Common runtime config shared by current built-in algorithms."""

    kl_coef: float = 0.01
    component_mix_stage: str = "reward"
    adv_normalization_scope: str = "group"
    samples_per_prompt: int = 1
    num_inference_steps: int = 0
    eval_ema_decay: float = 0.9
    eval_ema_update_interval: int = 1
    epsilon: float = 1e-8
    clip_max: Optional[float] = 5.0
    use_global_std: bool = False
    trim_outliers_ratio: float = 0.0


class BaseAlgorithm(ABC):
    """
    Base class for algorithm plugins.

    Each algorithm variant implements:
    - __init__(*, config: TypedConfig): Construct from typed runtime config
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - compute_loss(): Single algorithm-owned loss entrypoint
    - get_filtered_training_indices(): Optional timestep filtering for training

    The Algorithm class is the single source of truth for both rollout-side
    requirements (sampling, advantages) and training-side gradient computation.

    Training-side forward dispatch
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The ``ForwardContext`` (pure data container with model-specific fields)
    is carried on ``TrainingBatch.forward_context``. Subclasses read it
    directly from the batch and dispatch through the explicit
    ``model_bundle.forward_denoiser(...)`` interface owned by the active
    model family.
    """

    model_type: str = "default"
    __CONFIG_CLASS__ = BaseAlgorithmConfig

    def __init__(
        self,
        kl_coef: float = 0.01,
        component_mix_stage: str = "reward",
        adv_normalization_scope: str = "group",
        samples_per_prompt: int = 1,
        num_inference_steps: int = 0,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_global_std: bool = False,
        trim_outliers_ratio: float = 0.0,
        **kwargs,
    ):
        """
        Initialize algorithm.

        Args:
            kl_coef: KL penalty coefficient
            component_mix_stage: Multi-component reward mixing stage ("reward" or "advantage")
            adv_normalization_scope: Scope of advantage normalization ("global" or "group")
            samples_per_prompt: Number of rollout samples to generate per prompt
            num_inference_steps: Rollout denoising horizon shared by algorithm logic
            eval_ema_decay: Eval-time EMA decay
            eval_ema_update_interval: Eval-time EMA update interval in optimizer steps
            epsilon: Small value for numerical stability in advantage normalization
            clip_max: Maximum advantage value for clipping (None to disable)
            use_global_std: Use global std instead of per-group std
            trim_outliers_ratio: Ratio of outliers trimmed from each side for grouped stats
            **kwargs: Additional algorithm-specific arguments
        """
        self.kl_coef = kl_coef
        self.component_mix_stage = str(component_mix_stage)
        self.adv_normalization_scope = adv_normalization_scope
        self.samples_per_prompt = max(1, int(samples_per_prompt))
        self.num_inference_steps = max(0, int(num_inference_steps))
        self.eval_ema_decay = float(eval_ema_decay)
        self.eval_ema_update_interval = max(1, int(eval_ema_update_interval))
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.use_global_std = use_global_std
        self.trim_outliers_ratio = max(0.0, min(float(trim_outliers_ratio), 0.49))
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Class-level contracts (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_sampling_requirements(
        self,
    ) -> SamplingRequirements:  # [PUBLIC-API → driver rollout runtime] rollout requirements
        """
        Return the sampling requirements for this algorithm.

        Returns:
            SamplingRequirements specifying what the sampler needs to provide
        """
        ...

    def compute_advantages(  # [PUBLIC-API → rollout pipeline via compute_advantages_with_components()] rollout side
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards.

        Dispatches to the appropriate normalization method based on
        ``self.adv_normalization_scope`` ("global" or "group").

        Args:
            rewards: Reward tensor [batch_size]
            group_ids: Optional explicit sample-group identifiers for batch grouping

        Returns:
            Advantage tensor [batch_size]
        """
        if self.adv_normalization_scope == "global":
            return self._normalize_global(rewards)
        else:
            return self._normalize_group(
                rewards,
                group_ids=group_ids,
            )

    @staticmethod
    def _normalize_group_id(group_id: Any) -> Optional[str]:  # [HELPER]
        if group_id is None:
            return None
        text = str(group_id).strip()
        return text if text else None

    def _require_valid_group_ids(self, group_ids: List[str]) -> List[str]:  # [HELPER → _normalize_group()]
        normalized: List[str] = []
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                raise ValueError(
                    "adv_normalization_scope='group' requires a non-empty group_id for every sample. "
                    f"Found invalid group_id at sample_idx={sample_idx}."
                )
            normalized.append(group_id)
        return normalized

    def _build_group_index_map(self, group_ids: List[str]) -> Dict[str, List[int]]:  # [HELPER → _normalize_group()]
        ordered_groups: Dict[str, List[int]] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                continue
            ordered_groups.setdefault(group_id, []).append(sample_idx)
        return ordered_groups

    def _require_expected_group_sizes(
        self,
        group_index_map: Dict[str, List[int]],
    ) -> List[List[int]]:
        expected_group_size = max(1, int(self.samples_per_prompt))
        invalid_groups = [
            (group_id, len(indices))
            for group_id, indices in group_index_map.items()
            if len(indices) != expected_group_size
        ]
        if invalid_groups:
            formatted = ", ".join(f"{group_id!r}:{group_size}" for group_id, group_size in invalid_groups[:5])
            if len(invalid_groups) > 5:
                formatted = f"{formatted}, ..."
            raise ValueError(
                "adv_normalization_scope='group' requires every sample group to contain exactly "
                "algorithm.samples_per_prompt samples. "
                f"Got algorithm.samples_per_prompt={expected_group_size}; "
                f"invalid group sizes: {formatted}."
            )
        return list(group_index_map.values())

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:  # [HELPER → compute_advantages()]
        """Global normalization across all samples."""
        return normalize_global(rewards, epsilon=self.epsilon, clip_max=self.clip_max)

    def _normalize_group(  # [HELPER → compute_advantages()]
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Group normalization using explicit sample-group identities."""
        batch_size = rewards.shape[0]
        if group_ids is not None and len(group_ids) == batch_size:
            normalized_group_ids = self._require_valid_group_ids(group_ids)
            group_index_map = self._build_group_index_map(normalized_group_ids)
            groups = self._require_expected_group_sizes(group_index_map)
            if groups:
                return normalize_grouped(
                    rewards,
                    groups,
                    epsilon=self.epsilon,
                    clip_max=self.clip_max,
                    trim_outliers_ratio=self.trim_outliers_ratio,
                    use_global_std=self.use_global_std,
                )

        raise ValueError(
            "adv_normalization_scope='group' requires explicit group_ids aligned to the reward batch. "
            f"Got batch_size={batch_size}, "
            f"group_ids_len={len(group_ids) if group_ids is not None else None}."
        )

    # ------------------------------------------------------------------
    # Phase 3: Advantage orchestration with reward components
    # ------------------------------------------------------------------

    def compute_advantages_with_components(  # [PUBLIC-API → RolloutActor reward pipeline] rollout side
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        reward_components: Optional[Dict[str, List[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Full advantage computation pipeline including reward component mixing.

        This consolidates the logic that was previously scattered in rollout
        workflow helpers into the algorithm itself.

        Args:
            rewards: Aggregated reward tensor [batch_size].
            reward_components: Per-component reward values, keyed by name.
            reward_component_weights: Per-component weights for aggregation.

        Returns:
            Advantage tensor [batch_size].
        """
        if self.component_mix_stage != "advantage" or not reward_components:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)

        default_weights = {name: 1.0 for name in reward_components}
        if reward_component_weights:
            for name, weight in reward_component_weights.items():
                if name in default_weights:
                    default_weights[name] = float(weight)

        weighted_advantages = torch.zeros_like(rewards)
        total_weight = 0.0
        for component_name, component_rewards in reward_components.items():
            component_tensor = torch.tensor(
                component_rewards,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            if component_tensor.shape != rewards.shape:
                logger.warning(
                    "Skipping reward component %s due to shape mismatch: expected=%s got=%s",
                    component_name,
                    tuple(rewards.shape),
                    tuple(component_tensor.shape),
                )
                continue
            component_advantages = self.compute_advantages(
                rewards=component_tensor,
                group_ids=group_ids,
            )
            weight = float(default_weights.get(component_name, 1.0))
            weighted_advantages += component_advantages * weight
            total_weight += weight

        if total_weight <= 0:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)
        return weighted_advantages / total_weight

    @abstractmethod
    def get_ema_spec(
        self,
    ) -> EMASpec:  # [PUBLIC-API → TrainActor] training side
        """Declare EMA policy for this algorithm."""
        ...

    def prepare_loss_advantages(  # [PUBLIC-API → TrainStack] training side advantage transform
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Transform rollout advantages into the signal consumed by the loss.

        Default behavior is identity. Algorithms such as NFT can override this
        hook to map rollout advantages into loss-side weighting signals without
        mutating the rollout-visible advantage semantics.
        """
        return advantages

    # ------------------------------------------------------------------
    # Phase 2: Algorithm-owned training step
    # ------------------------------------------------------------------

    def compute_loss_and_backward(  # [PUBLIC-API → TrainStack] training side: core loss+backward
        self,
        *,
        model_bundle: ModelBundle,
        batch: Any,
        timesteps: Any = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """Compute loss and call backward for a single micro-batch.

        The micro-batch loop is owned by ``TrainStack``; each algorithm
        only needs to handle one already-sliced micro-batch here.

        Args:
            model_bundle: The model bundle to compute loss on.
            batch: A single micro-batch (already sliced by TrainStack).
            timesteps: Pre-resolved training timesteps from
                ``resolve_training_timesteps()``.
            loss_scale: Gradient accumulation scale (1 / num_mini_batches).

        Returns:
            ``(loss, metrics, num_timesteps, has_backward)``
        """
        del model_bundle, batch, timesteps, loss_scale, kwargs
        raise NotImplementedError(f"{type(self).__name__} must implement compute_loss_and_backward().")

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        """Resolve algorithm-defined training timesteps once per update chunk.

        The returned object should represent actual training timesteps for the
        update. Algorithms that internally train by logical step index may map
        those timesteps back to indices inside ``compute_loss_and_backward``.
        """
        del batch, current_step, kwargs
        return None

    def is_forward_process(
        self,
    ) -> bool:  # [PUBLIC-API → TrainActor] training side algorithm type
        """Whether this algorithm trains via forward process (NFT-style)."""
        return bool(self.get_sampling_requirements().is_forward_process)

    @abstractmethod
    def resolve_rollout_sde_indices(  # [PUBLIC-API → train.py rollout orchestration] rollout side
        self,
        *,
        current_step: int,
    ) -> Set[int]:
        """Resolve rollout-time SDE indices for the current step.

        Default behavior:
        - Forward-process algorithms do not use rollout SDE indices.
        - Trajectory algorithms implement their own current-step-driven policy.
        """
        ...

    @abstractmethod
    def get_sampler_validation_config(
        self, *, allow_replay: bool
    ) -> Dict[str, Any]:  # [PUBLIC-API → driver rollout pipeline] rollout side
        """Get sampler-output validation flags for rollout orchestration."""
        ...

    def get_filtered_training_indices(  # [PUBLIC-API → RolloutPipeline.convert_training_data] rollout side: filter training timesteps
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        """
        Get filtered training timestep indices based on algorithm requirements.

        This method handles common filtering operations like:
        - skip_last_timestep: Skip the last timestep (t->0) which has unstable log_prob
        - skip_initial_timesteps: Skip early timesteps with high variance

        Subclasses can override to add algorithm-specific filtering.
        Default returns *sde_indices* unchanged.

        Args:
            sde_indices: Set of SDE timestep indices from scheduler
            num_steps: Total number of timesteps

        Returns:
            Filtered set of timestep indices for training
        """
        return sde_indices

    def get_config(self) -> Dict[str, Any]:  # [PUBLIC-API → serialization/logging]
        """Get algorithm configuration as dictionary."""
        return {
            "algorithm_type": self.__class__.__name__,
            "kl_coef": self.kl_coef,
            "component_mix_stage": self.component_mix_stage,
            "adv_normalization_scope": self.adv_normalization_scope,
            **self._extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"kl_coef={self.kl_coef}, "
            f"component_mix_stage={self.component_mix_stage}, "
            f"adv_normalization_scope={self.adv_normalization_scope})"
        )
