"""Driver-side rollout-time controller for GRPO.

The trainer-side :class:`StageAlgorithm` (``algorithms_new.DiffusionGRPO`` /
``ARGRPO``) handles per-slot loss + backward. The driver-side controller
picks WHICH inference timesteps run SDE per rollout step, and filters
which timesteps actually enter the training batch. These are independent
concerns; the recipe convention keeps them under two distinct keys
(singular ``cfg.algorithm`` for driver-side, plural ``cfg.algorithms.<slot>``
for trainer-side).

Wired by ``train_new.py``::

    control_algorithm = build(cfg.algorithm)
    rollout_pipeline.plan_requests(
        ..., control_algorithm=control_algorithm
    )

Consumed surface (from ``rollout/new_pipeline.py`` + ``train_new.py``):

- :meth:`resolve_rollout_sde_indices(current_step)` — per rollout step
  decides which inference indices run SDE (called by
  :meth:`NewRolloutPipeline.plan_requests`).
- :meth:`get_filtered_training_indices(sde_indices, num_steps)` — filters
  training timesteps (called by
  :meth:`NewRolloutPipeline.convert_training_data`).
- :attr:`samples_per_prompt` — read by ``train_new.py`` for rollout-batch
  sizing.

Schema fields shared between the singular ``cfg.algorithm`` block here and
the plural ``cfg.algorithms.<slot>`` trainer-side configs are kept aligned
so a single recipe can drive both ends without duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set

import torch
from omegaconf import SI

from diffusionrl.algorithms_new.normalizers import normalize_global, normalize_grouped
from diffusionrl.config.registration import register_config
from diffusionrl.types.sampling import SamplingParams, SDEConfig
from diffusionrl.utils.scheduler_utils import SchedulerConfig, create_indices_scheduler


# Declarative schema for the ``ema:`` block under ``cfg.algorithm``;
# recipes carry these fields so OmegaConf structured-config merge
# succeeds. The actual EMA mechanics live on the trainer-side
# ``EMAPolicy`` / ``NFTLoRAPolicy`` — this dataclass is recipe data
# only.
@dataclass
class GRPOEMAConfig:
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


@register_config(
    group="algorithm",
    name="grpo",
    target="diffusionrl.algorithms_new.rollout_control.GRPORolloutControl",
    mutable=True,
)
@dataclass
class GRPORolloutControlConfig:
    """Driver-side GRPO controller config.

    Fields actually READ by :class:`GRPORolloutControl`:

    - ``sampling.num_inference_steps`` (via SI-interpolated
      ``${sampling}``)
    - ``samples_per_prompt``
    - ``prompts_per_rollout`` (logged)
    - ``scheduler`` (built into :attr:`rollout_indices_scheduler`)
    - ``skip_last_timestep``, ``skip_initial_timesteps``

    Fields carried for recipe-shape compat (consumed by the plural
    ``cfg.algorithms.<slot>`` :class:`StageAlgorithm` configs, not here):

    - ``adv_normalization_scope``, ``use_global_std`` (advantage policy)
    - ``clip_range``, ``clip_schedule`` (PPO clipping)
    - ``use_kl_penalty``, ``kl_coef`` (KL penalty — see __post_init__)
    - ``sde_config`` (SDE replay eta)
    - ``ema`` (eval-EMA policy)
    """

    sampling: SamplingParams = field(default_factory=lambda: SI("${sampling}"))

    prompts_per_rollout: int = 1
    samples_per_prompt: int = 1

    # Driver-side scheduler — USED
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Driver-side training-timestep filter — USED
    skip_last_timestep: bool = False
    skip_initial_timesteps: int = 0

    # Recipe compat — carried but not read by GRPORolloutControl.
    # The plural cfg.algorithms.<slot> StageAlgorithm consumers own these.
    adv_normalization_scope: str = "group"
    use_global_std: bool = False
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    # The KL penalty against the reference policy is not implemented in
    # this revision; ``__post_init__`` rejects ``kl_coef > 0`` so a recipe
    # cannot silently lose the term. ``use_kl_penalty`` is preserved as a
    # historical recipe knob but has no runtime effect (its intent is now
    # encoded entirely in ``kl_coef``).
    use_kl_penalty: bool = True
    kl_coef: float = 0.0
    sde_config: SDEConfig = field(default_factory=SDEConfig)
    ema: GRPOEMAConfig = field(default_factory=GRPOEMAConfig)

    def __post_init__(self) -> None:
        if float(self.kl_coef) > 0:
            raise ValueError(
                f"GRPORolloutControlConfig.kl_coef={self.kl_coef!r}: the KL "
                f"penalty against the reference policy is not implemented. "
                f"Set kl_coef=0 (the recipe default) or implement the term."
            )


class GRPORolloutControl:
    """Driver-side rollout-time scheduler for GRPO.

    Class flag ``_REQUIRES_SDE_STEPS`` controls whether the constructor
    rejects ``scheduler.num_sde_steps=0``. Subclasses that don't need
    SDE steps (NFT) flip it to ``False``.
    """

    #: Reject ``scheduler.num_sde_steps=0`` at construction. Forward-
    #: process subclasses (NFT) override to ``False``.
    _REQUIRES_SDE_STEPS: ClassVar[bool] = True

    def __init__(self, *, config: GRPORolloutControlConfig, **kwargs):
        del kwargs  # tolerate extras Hydra may pass via build()
        if not isinstance(config, GRPORolloutControlConfig):
            raise TypeError(f"{type(self).__name__} expects GRPORolloutControlConfig, got {type(config).__name__}.")
        # ``AllSDEScheduler`` accepts ``num_sde_steps=0`` for forward-process
        # algorithms (NFT). GRPO requires SDE steps to train — without
        # them the segment has no SDE log-probs and the algorithm sees
        # ``sde_indices=None`` → ``has_backward=False`` silently. Reject
        # the misconfiguration explicitly here; NFTRolloutControl flips
        # ``_REQUIRES_SDE_STEPS=False`` and is the only legal consumer
        # of ``num_sde_steps=0``.
        num_sde_steps = getattr(config.scheduler, "num_sde_steps", None)
        if self._REQUIRES_SDE_STEPS and num_sde_steps is not None and int(num_sde_steps) == 0:
            raise ValueError(
                f"{type(self).__name__}: scheduler.num_sde_steps=0 is only "
                f"valid for forward-process algorithms (NFTRolloutControl). "
                f"GRPO requires SDE steps to train; either set num_sde_steps "
                f"> 0 (or leave it null to use every step in the fraction "
                f"range), or switch the recipe to algorithm=nft."
            )
        self.config = config
        self.sampling = config.sampling
        self.prompts_per_rollout = int(config.prompts_per_rollout)
        self.samples_per_prompt = max(1, int(config.samples_per_prompt))
        self.skip_last_timestep = bool(config.skip_last_timestep)
        self.skip_initial_timesteps = int(config.skip_initial_timesteps)
        self.rollout_indices_scheduler = create_indices_scheduler(
            scheduler_config=config.scheduler,
            num_timesteps=self.sampling.num_inference_steps,
        )

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Optional[Set[int]]:
        if self.sampling.num_inference_steps < 1:
            raise ValueError(
                f"{type(self).__name__}.resolve_rollout_sde_indices requires "
                f"sampling.num_inference_steps >= 1, got "
                f"{self.sampling.num_inference_steps}."
            )
        return set(self.rollout_indices_scheduler.get_sde_indices(current_step))

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        result = set(sde_indices)
        if self.skip_last_timestep and result:
            result.discard(max(result))
        if self.skip_initial_timesteps > 0:
            result = {i for i in result if i >= self.skip_initial_timesteps}
        return result

    # ------------------------------------------------------------------
    # Advantage computation — rollout-side, called from
    # ``ray/mixins/new_rollout_pipeline.py`` (compute_advantages,
    # run_rollout_pipeline). ``epsilon`` / ``clip_max`` /
    # ``trim_outliers_ratio`` are class-level defaults; expose them on
    # :class:`GRPORolloutControlConfig` if a recipe ever needs to tune them.
    # ------------------------------------------------------------------

    _ADV_EPSILON: ClassVar[float] = 1e-8
    _ADV_CLIP_MAX: ClassVar[Optional[float]] = None
    _ADV_TRIM_OUTLIERS_RATIO: ClassVar[float] = 0.0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute advantages from rewards.

        Dispatches to global or group normalization based on
        ``self.config.adv_normalization_scope``.
        """
        scope = str(self.config.adv_normalization_scope)
        if scope == "global":
            return self._normalize_global(rewards)
        if scope == "group":
            return self._normalize_group(rewards, group_ids=group_ids)
        raise ValueError(f"Unknown adv_normalization_scope={scope!r}. Expected 'global' or 'group'.")

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:
        return normalize_global(
            rewards,
            epsilon=self._ADV_EPSILON,
            clip_max=self._ADV_CLIP_MAX,
        )

    def _normalize_group(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        if group_ids is None or len(group_ids) != batch_size:
            raise ValueError(
                "adv_normalization_scope='group' requires explicit group_ids "
                f"aligned to the reward batch. Got batch_size={batch_size}, "
                f"group_ids_len={len(group_ids) if group_ids is not None else None}."
            )
        normalized_group_ids = self._require_valid_group_ids(group_ids)
        group_index_map = self._build_group_index_map(normalized_group_ids)
        groups = self._require_expected_group_sizes(group_index_map)
        if not groups:
            raise ValueError(
                "adv_normalization_scope='group' could not find any valid group; "
                "all group_ids were empty after normalization."
            )
        return normalize_grouped(
            rewards,
            groups,
            epsilon=self._ADV_EPSILON,
            clip_max=self._ADV_CLIP_MAX,
            trim_outliers_ratio=self._ADV_TRIM_OUTLIERS_RATIO,
            use_global_std=bool(self.config.use_global_std),
        )

    @staticmethod
    def _normalize_group_id(group_id: Any) -> Optional[str]:
        if group_id is None:
            return None
        text = str(group_id).strip()
        return text if text else None

    def _require_valid_group_ids(self, group_ids: List[str]) -> List[str]:
        normalized: List[str] = []
        for sample_idx, raw_group_id in enumerate(group_ids):
            gid = self._normalize_group_id(raw_group_id)
            if gid is None:
                raise ValueError(
                    "adv_normalization_scope='group' requires a non-empty group_id "
                    f"for every sample. Invalid group_id at sample_idx={sample_idx}."
                )
            normalized.append(gid)
        return normalized

    def _build_group_index_map(self, group_ids: List[str]) -> Dict[str, List[int]]:
        ordered_groups: Dict[str, List[int]] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            gid = self._normalize_group_id(raw_group_id)
            if gid is None:
                continue
            ordered_groups.setdefault(gid, []).append(sample_idx)
        return ordered_groups

    def _require_expected_group_sizes(
        self,
        group_index_map: Dict[str, List[int]],
    ) -> List[List[int]]:
        expected = max(1, int(self.samples_per_prompt))
        invalid = [(gid, len(idxs)) for gid, idxs in group_index_map.items() if len(idxs) != expected]
        if invalid:
            formatted = ", ".join(f"{gid!r}:{size}" for gid, size in invalid[:5])
            if len(invalid) > 5:
                formatted = f"{formatted}, ..."
            raise ValueError(
                "adv_normalization_scope='group' requires every sample group to contain exactly "
                f"samples_per_prompt={expected} samples. Invalid group sizes: {formatted}."
            )
        return list(group_index_map.values())


@register_config(
    group="algorithm",
    name="nft",
    target="diffusionrl.algorithms_new.rollout_control.NFTRolloutControl",
    mutable=True,
)
@dataclass
class NFTRolloutControlConfig(GRPORolloutControlConfig):
    """NFT driver-side config. Schema identical to
    :class:`GRPORolloutControlConfig`; the difference is purely behavioral
    and lives in :class:`NFTRolloutControl` (no SDE steps, no training-index
    filtering). Sharing the schema keeps recipe shape stable across
    GRPO ↔ NFT swap (``defaults: - override /algorithm: nft``).
    """


class NFTRolloutControl(GRPORolloutControl):
    """Forward-process driver-side rollout controller.

    Two methods diverge from :class:`GRPORolloutControl`:

    * :meth:`resolve_rollout_sde_indices` returns ``None``. The rollout
      engines (vllm-omni, sglang) read this as "no step runs SDE; no
      log-prob captured" and follow their forward-process branch.
    * :meth:`get_filtered_training_indices` returns an empty set. NFT
      picks its own timesteps inside the loss; the segment's SDE
      indices are not consumed.

    Advantage normalization is inherited unchanged — NFT and GRPO use
    the same group-relative formula.
    """

    # NFT recipes set ``scheduler.num_sde_steps=0`` so the rollout engine
    # takes the forward-process path. Bypass the GRPO check that rejects
    # this value (GRPO would silently no-train otherwise).
    _REQUIRES_SDE_STEPS: ClassVar[bool] = False

    def __init__(self, *, config: GRPORolloutControlConfig, **kwargs):
        # Accept either NFTRolloutControlConfig or its GRPO parent (schema
        # is identical). Hydra instantiation passes the registered
        # NFTRolloutControlConfig; programmatic construction may pass
        # the base for convenience.
        super().__init__(config=config, **kwargs)

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Optional[Set[int]]:
        # ``None`` (not an empty set) lets vllm-omni / sglang short-circuit
        # the SDE branch entirely. Empty set means "schedule chose to skip
        # everything for this rollout" which is a different signal.
        return None

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        # NFT picks its own ``t`` inside ``DiffusionNFT.compute_loss_and_backward``;
        # the segment's SDE indices (if any) are irrelevant here.
        return set()


__all__ = [
    "GRPOEMAConfig",
    "GRPORolloutControl",
    "GRPORolloutControlConfig",
    "NFTRolloutControl",
    "NFTRolloutControlConfig",
]
