"""Stage-driven algorithm base class.

The training-side contract for ``models_new`` pipelines: an algorithm
holds a stage (``DiffusionStage[C]`` or ``ARStage[C]``) and computes loss
over ``(conditions, segment, advantages)``. All model dispatch, CFG batching,
SDE math, autocast, and per-step iteration are owned by ``stage.replay(...)``;
the algorithm is pure ratio-clip math against the segment's stored log-probs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

if TYPE_CHECKING:
    from diffusionrl.types.conditions import Condition
    from diffusionrl.types.segments.base import Segment


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


class StageAlgorithm(ABC):
    """Pure (conditions, segment, advantages) → loss; holds its stage.

    Targets the four-tier pipeline contract (``models_new``). The algorithm
    holds a reference to a
    :class:`diffusionrl.models_new.types.diffusion.DiffusionStage` or
    :class:`diffusionrl.models_new.types.ar.ARStage` and dispatches all
    model forward / SDE / CFG work into ``stage.replay(...)``. It does not
    know its slot key in the dispatcher; slot routing lives on the train stack.
    """

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
        :meth:`diffusionrl.ray.new_train_actor.NewTrainActor._train_resp`
        — the populated tensor is frozen at pre-update weights across all
        N updates, matching the on-policy ratio semantics of PPO-style
        algorithms.

        Args:
            conditions: ``RolloutResp.conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.rollout_traces[slot]`` for this algorithm's
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
            conditions: ``RolloutResp.conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.rollout_traces[slot]`` — diffusion algorithms
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


__all__ = ["AlgorithmStepResult", "StageAlgorithm"]
