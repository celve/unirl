"""
diffusionrl Algorithm Base Class.

Defines algorithm responsibilities in rollout/advantage pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

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
    eval_decay: float = 0.9
    eval_update_interval: int = 1
    reference_mode: str = "none"
    reference_decay: float = 0.0
    reference_decay_type: str = "constant"
    reference_flat_steps: int = 0
    reference_uprate: float = 0.001
    reference_uphold: float = 0.5
    reference_update_timing: str = "optimizer_step"
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"


class BaseAlgorithm(ABC):
    """
    Base class for algorithm plugins.

    Each algorithm variant implements:
    - from_config(): Construct from algorithm_config dict (classmethod)
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - compute_loss(): Single algorithm-owned loss entrypoint
    - optional timestep filtering helpers for backward training assembly

    The Algorithm class is the single source of truth for both rollout-side
    requirements (sampling, advantages) and training-side gradient computation.
    """

    _loss_cls = None

    def __init__(
        self,
        kl_coef: float = 0.01,
        component_mix_stage: str = "reward",
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_global_std: bool = False,
        trimmed_ratio: float = 0.0,
        **kwargs,
    ):
        """
        Initialize algorithm.

        Args:
            kl_coef: KL penalty coefficient
            component_mix_stage: Multi-component reward mixing stage ("reward" or "advantage")
            adv_normalization: Type of advantage normalization ("global" or "group")
            samples_per_prompt: Number of rollout samples to generate per prompt
            eval_ema_decay: Eval-time EMA decay
            eval_ema_update_interval: Eval-time EMA update interval in optimizer steps
            epsilon: Small value for numerical stability in advantage normalization
            clip_max: Maximum advantage value for clipping (None to disable)
            use_global_std: Use global std instead of per-group std
            trimmed_ratio: Ratio of outliers trimmed from each side for grouped stats
            **kwargs: Additional algorithm-specific arguments
        """
        self.kl_coef = kl_coef
        self.component_mix_stage = str(component_mix_stage)
        self.adv_normalization = adv_normalization
        self.samples_per_prompt = max(1, int(samples_per_prompt))
        self.eval_ema_decay = float(eval_ema_decay)
        self.eval_ema_update_interval = max(1, int(eval_ema_update_interval))
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.use_global_std = use_global_std
        self.trimmed_ratio = max(0.0, min(float(trimmed_ratio), 0.49))
        self._extra_kwargs = kwargs

        self.loss_fn = self._create_loss_fn()

    def _create_loss_fn(self):
        loss_cls = getattr(type(self), "_loss_cls", None)
        if loss_cls is None:
            return None
        return loss_cls(self)

    # ------------------------------------------------------------------
    # Class-level contracts (override in subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "BaseAlgorithm":  # [PUBLIC-API → rollout_manager.init(), training_actor.init()]
        """Construct algorithm from an algorithm_config dictionary.

        TrainingActor calls ``algorithm_cls.from_config(algorithm_config)`` to
        create the train-side algorithm instance.  Subclasses should override
        to read their specific parameters from ``config`` and
        ``config['algorithm_kwargs']``.

        Default implementation raises NotImplementedError so custom
        plugins fail loudly if they forget to implement this.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config() classmethod."
        )

    @abstractmethod
    def get_sampling_requirements(self) -> SamplingRequirements:  # [PUBLIC-API → rollout_manager.init()] 推理侧: 声明采样需求
        """
        Return the sampling requirements for this algorithm.

        Returns:
            SamplingRequirements specifying what the sampler needs to provide
        """
        ...

    def compute_advantages(  # [PUBLIC-API → rollout_manager via compute_advantages_with_components()] 推理侧
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards.

        Dispatches to the appropriate normalization method based on
        ``self.adv_normalization`` ("global" or "group").

        Args:
            rewards: Reward tensor [batch_size]
            group_ids: Optional explicit sample-group identifiers for batch grouping

        Returns:
            Advantage tensor [batch_size]
        """
        if self.adv_normalization == "global":
            return self._normalize_global(rewards)
        elif self.adv_normalization == "group":
            return self._normalize_group(
                rewards,
                group_ids=group_ids,
            )
        else:
            raise ValueError(f"Unknown adv_normalization: {self.adv_normalization}")

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
                    "adv_normalization='group' requires a non-empty group_id for every sample. "
                    f"Found invalid group_id at sample_idx={sample_idx}."
                )
            normalized.append(group_id)
        return normalized

    def _build_groups_from_ids(self, group_ids: List[str]) -> List[List[int]]:  # [HELPER → _normalize_group()]
        ordered_groups: Dict[str, List[int]] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                continue
            ordered_groups.setdefault(group_id, []).append(sample_idx)
        return list(ordered_groups.values())

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
            groups = self._build_groups_from_ids(normalized_group_ids)
            if groups:
                return normalize_grouped(
                    rewards,
                    groups,
                    epsilon=self.epsilon,
                    clip_max=self.clip_max,
                    trimmed_ratio=self.trimmed_ratio,
                    use_global_std=self.use_global_std,
                )

        raise ValueError(
            "adv_normalization='group' requires explicit group_ids aligned to the reward batch. "
            f"Got batch_size={batch_size}, "
            f"group_ids_len={len(group_ids) if group_ids is not None else None}."
        )

    # ------------------------------------------------------------------
    # Phase 3: Advantage orchestration with reward components
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_advantages_with_components(  # [PUBLIC-API → rollout_workflow.compute_advantages()] 推理侧
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
        ...

    @abstractmethod
    def get_ema_spec(self) -> EMASpec:  # [PUBLIC-API → training_actor.init()] 训练侧
        """Declare EMA policy for this algorithm."""
        ...

    def prepare_loss_advantages(  # [PUBLIC-API → train_executor] 训练侧: loss 前 advantage 变换
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

    def compute_loss_and_backward(  # [PUBLIC-API → train_executor._train_update_chunk()] 训练侧: 核心 loss+backward
        self,
        *,
        model: nn.Module,
        batch: Any,
        mini_batch_slices: Tuple[Tuple[int, int], ...],
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> tuple:
        """Compute loss and call backward for a single update chunk."""
        del model, batch, mini_batch_slices, guidance_scale, kwargs
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_loss_and_backward()."
        )

    def is_forward_process(self) -> bool:  # [PUBLIC-API → training_actor] 训练侧: 判断算法类型
        """Whether this algorithm trains via forward process (NFT-style)."""
        return bool(self.get_sampling_requirements().is_forward_process)

    @abstractmethod
    def resolve_rollout_sde_indices(  # [PUBLIC-API → train.py RolloutSDEController] 推理侧
        self,
        *,
        timestep_scheduler: Optional[Any],
        current_step: int,
    ) -> Optional[Set[int]]:
        """Resolve rollout-time SDE indices for the current step.

        Default behavior:
        - Forward-process algorithms do not use rollout SDE indices.
        - Trajectory algorithms read indices from the caller-provided scheduler.
        """
        ...

    @abstractmethod
    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:  # [PUBLIC-API → rollout_manager] 推理侧
        """Get sampler-output validation flags for rollout orchestration."""
        ...

    @abstractmethod
    def assemble_training_batch(  # [PUBLIC-API → rollout_workflow.build_training_batch()] 推理侧: 组装 TrainingBatch
        self,
        *,
        num_inference_steps: int,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        """Assemble the typed training batch for this algorithm."""
        ...

    def resolve_training_indices(
        self,
        *,
        num_steps: int,
        sde_indices: Optional[Set[int]] = None,
    ) -> Set[int]:
        """Resolve the timestep indices that should contribute to training.

        This is the explicit counterpart to rollout-time ``sde_indices``.
        """
        if sde_indices is not None:
            return set(int(i) for i in sde_indices)
        return set(range(num_steps))

    @abstractmethod
    def get_filtered_training_indices(  # [PUBLIC-API → assemble_training_batch() 内部] 推理侧: 过滤训练 timestep
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

        Args:
            sde_indices: Set of SDE timestep indices from scheduler
            num_steps: Total number of timesteps

        Returns:
            Filtered set of timestep indices for training
        """
        ...

    def get_config(self) -> Dict[str, Any]:  # [PUBLIC-API → 序列化/日志]
        """Get algorithm configuration as dictionary."""
        return {
            "algorithm_type": self.__class__.__name__,
            "kl_coef": self.kl_coef,
            "component_mix_stage": self.component_mix_stage,
            "adv_normalization": self.adv_normalization,
            **self._extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"kl_coef={self.kl_coef}, "
            f"component_mix_stage={self.component_mix_stage}, "
            f"adv_normalization={self.adv_normalization})"
        )
