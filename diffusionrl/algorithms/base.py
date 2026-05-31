"""Stage-driven algorithm base class.

The training-side contract for ``models`` pipelines: an algorithm
holds a stage (``DiffusionStage[C]`` or ``ARStage[C]``) and computes loss
over ``(conditions, segment, advantages)``. All model dispatch, CFG batching,
SDE math, autocast, and per-step iteration are owned by ``stage.replay(...)``;
the algorithm is pure ratio-clip math against the segment's stored log-probs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Type

import torch

from diffusionrl.distributed.group.remote import Remote

if TYPE_CHECKING:
    from diffusionrl.types.conditions import Condition
    from diffusionrl.types.segments.base import Segment


# ---------------------------------------------------------------------------
# Shared helpers used by DiffusionGRPO, DiffusionDPPO, DiffusionNFT
# ---------------------------------------------------------------------------


def typed_conditions(
    conditions: Mapping[str, "Condition"],
    conditions_cls: Optional[Type[Any]],
) -> Any:
    """Reconstruct the stage's typed conditions container from the dict shape.

    When ``conditions_cls`` is ``None`` (e.g. unit tests against a fake stage
    that accepts the dict directly), the dict is forwarded verbatim. Otherwise
    ``conditions_cls.from_dict(...)`` is invoked.
    """
    if conditions_cls is None:
        return conditions
    return conditions_cls.from_dict(dict(conditions))


def gather_sde_field(
    tensor: Optional[torch.Tensor],
    sde_indices: Optional[torch.Tensor],
    target_steps: List[int],
    *,
    field_name: str = "field",
) -> torch.Tensor:
    """Gather slices from a segment's SDE-aligned tensor by step index.

    Maps ``target_steps`` to positions in ``sde_indices`` via
    ``torch.searchsorted`` (O(S' log S)) and returns
    ``tensor[:, positions, ...]``.

    Used by GRPO (for ``sde_logp``) and DPPO (for ``sde_logp`` + ``sde_means``).
    """
    if tensor is None or sde_indices is None:
        raise ValueError(
            f"gather_sde_field: {field_name} or sde_indices is None "
            f"(ensure prepare_segment ran before compute_loss_and_backward)."
        )
    target_t = torch.tensor(target_steps, dtype=sde_indices.dtype, device=sde_indices.device)
    # Ensure sde_indices is sorted (searchsorted requirement)
    sort_order = sde_indices.argsort()
    sde_indices = sde_indices[sort_order]
    tensor = tensor[:, sort_order.tolist()]
    positions = torch.searchsorted(sde_indices, target_t)
    # Clamp to valid range before validation (searchsorted can return len for out-of-range)
    positions = positions.clamp(max=sde_indices.shape[0] - 1)
    # Validate looked-up positions match
    if (sde_indices[positions] != target_t).any():
        bad = [int(t) for t, p in zip(target_steps, positions) if sde_indices[p] != t]
        raise ValueError(
            f"gather_sde_field({field_name}): target steps {bad} not in sde_indices={sde_indices.tolist()}"
        )
    return tensor[:, positions.tolist()]


def rollout_replay_logp_absdiff(new_logp: torch.Tensor, old_logp: torch.Tensor) -> Dict[str, float]:
    """Per-token |Δlogp| between rollout and replay — AR train-rollout drift gauge.

    ``old_logp`` is the rollout-time log-prob (SGLang / trainside autoregress)
    and ``new_logp`` is the teacher-forced replay at the current weights. On a
    single on-policy update the two differ only by the rollout-vs-replay *engine*
    gap (a temperature/logprob misconfig, a broken SGLang weight sync, or bf16
    KV-cache-vs-full-forward drift). ``mean|Δlogp|`` reports that gap directly and
    symmetrically — more legible than the exp-biased ``ratio_mean``. AR-only: the
    diffusion algorithms self-record or recompute ``old_logp`` with the same
    model, so their gap is ~0 by construction and they do not emit this metric.

    Assumes non-empty inputs, mirroring ``_grpo_clip_loss`` — the AR callers
    early-return on a zero-token segment before this runs.
    """
    with torch.no_grad():
        absdiff = (new_logp - old_logp).abs()
    return {
        "rollout_replay_logp_absdiff_mean": float(absdiff.mean()),
        "rollout_replay_logp_absdiff_max": float(absdiff.max()),
    }


@dataclass(frozen=True)
class AlgorithmStepResult:
    """Result of one micro-step under the stage-driven contract.

    ``num_steps_or_tokens`` is the diffusion step count for diffusion
    algorithms or the trained-token count for AR algorithms.
    """

    loss: float
    metrics: Mapping[str, Any]
    num_steps_or_tokens: int
    has_backward: bool


class BaseAlgorithmConfig(ABC):
    """Marker base for all algorithm config dataclasses.

    Used as the type annotation for polymorphic algorithm config fields so that
    static type checkers see a meaningful type. At runtime, the annotation
    is erased to ``Any`` by ``erase_polymorphic_annotations``.
    """


class StageAlgorithm(Remote, ABC):
    """Pure (conditions, segment, advantages) → loss; holds its stage.

    Targets the four-tier pipeline contract (``models``). The algorithm
    holds a reference to a
    :class:`diffusionrl.models.types.diffusion.DiffusionStage` or
    :class:`diffusionrl.models.types.ar.ARStage` and dispatches all
    model forward / SDE / CFG work into ``stage.replay(...)``. It does not
    know its slot key in the dispatcher; slot routing lives on the train stack.

    Class attributes:
        requires_ema_rollout: Whether the algorithm requires EMA weights
            during rollout sampling. On-policy algorithms (GRPO) MUST
            sample with the same weights used in training replay so the
            importance ratio equals 1 on the first step (default False).
            Off-policy / forward-process algorithms (NFT) override to
            True so the rollout uses EMA-smoothed weights for higher-
            quality trajectories.
    """

    requires_ema_rollout: bool = False

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
    ) -> None:
        """Optional pre-step hook called once before the multi-update loop.

        Default no-op. Algorithms whose log-prob source is rollout-native
        (e.g. NFT, SFT, native-logprob GRPO) can ignore the hook entirely.

        Algorithms that need to lazily populate segment fields before
        training override this. The canonical use case is
        :class:`DiffusionGRPO` running in SGLang ``logprob_source='replay'``
        mode: SGLang emits the trajectory but not the per-step log-probs,
        so the trainer fills ``segment.sde_logp`` here via a ``torch.no_grad``
        ``stage.replay``. Because this hook fires ONCE per ``RolloutResp``
        — before the ``num_updates_per_batch`` loop in
        :meth:`diffusionrl.ray.train_actor.TrainActor._train_resp`
        — the populated tensor is frozen at pre-update weights across all
        N updates, matching the on-policy ratio semantics of PPO-style
        algorithms.

        Args:
            conditions: ``RolloutResp.tracks[slot].conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.tracks[slot].segment`` for this algorithm's
                slot. Implementations may mutate field defaults that were
                left ``None`` by the rollout (lazy initialization); they
                must NOT mutate fields that the rollout already populated.
        """
        return None

    @abstractmethod
    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Compute loss for one micro-batch and call ``.backward()``.

        Args:
            conditions: ``RolloutResp.tracks[slot].conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.tracks[slot].segment`` — diffusion algorithms
                read ``segment.sde_logp`` / ``segment.sde_indices`` /
                ``segment.sigmas``; AR algorithms read ``segment.log_probs`` /
                ``segment.cu_seqlens``.
            advantages: per-sample advantage signal ``[B]``.
            training_progress: training progress in ``[0, 1]`` for
                clip-range or other schedules.
            loss_scale: gradient accumulation factor (typically
                ``1 / num_micro_batches``).
        """
        ...


__all__ = [
    "AlgorithmStepResult",
    "StageAlgorithm",
    "gather_sde_field",
    "rollout_replay_logp_absdiff",
    "typed_conditions",
]
