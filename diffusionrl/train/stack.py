"""Single-stage train stack.

Wraps one :class:`FSDPBackend` (training state: model + optimizer +
scheduler + EMA) and one :class:`StageAlgorithm` (loss + backward
against the bundle's trainable module) into a single-stage training
driver.  One :class:`TrainStack` = one training track.

Sequencing per :meth:`train` call::

    backend.zero_grad()
    for (start, end) in micro_slices(resp_track.batch_size):
        algorithm.compute_loss_and_backward(loss_scale=1/N, ...)
    if has_backward:
        grad_norm = backend.optimizer_step(max_grad_norm=...)
    return TrainStepResult(loss, grad_norm, lr, has_backward, micros, metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Tuple

import torch

from diffusionrl.algorithms import AlgorithmStepResult, StageAlgorithm
from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote
from diffusionrl.distributed.tensor.batch import _move_value
from diffusionrl.train.backend.fsdp import FSDPBackend
from diffusionrl.types.rollout_resp import RolloutTrack
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainStepResult:
    """Result of one full optimizer step on this stage."""

    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    micros: List[AlgorithmStepResult]
    metrics: Mapping[str, object]


def _positive_int(*, name: str, value: object) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def _build_micro_batch_slices(
    *,
    total_size: int,
    micro_batch_size: int,
) -> Tuple[Tuple[int, int], ...]:
    resolved_total_size = _positive_int(name="total_size", value=total_size)
    resolved_micro_batch_size = _positive_int(name="micro_batch_size", value=micro_batch_size)
    slices: List[Tuple[int, int]] = []
    start = 0
    while start < resolved_total_size:
        end = min(start + resolved_micro_batch_size, resolved_total_size)
        slices.append((start, end))
        start = end
    return tuple(slices)


def _align_track_to_model(resp_track: RolloutTrack, *, device: torch.device) -> None:
    """Move a track's training inputs onto the model's device — SGLang returns
    them on CPU via Ray IPC. Uses :meth:`Batch.to_device` (recursive; carries
    framework-managed ``_packed_cu_seqlens`` and tensors nested in tuples/dicts)
    on the segment + conditions only, so heavy ``decoded`` / ``media_preview``
    payloads stay off the GPU. dtype is left to the model, which casts what it
    feeds the network (see SD3DiffusionStep.predict_noise).

    Condition values are moved via ``_move_value`` (the same recursive mover
    ``Batch.to_device`` uses) rather than assuming each value is a ``Batch``:
    most are (e.g. ``TextTokenCondition``), but multimodal stages also carry
    raw per-sample ``FieldKind.CONCAT`` lists of tensors (Qwen2.5-VL's
    ``pixel_values`` / ``image_grid_thw``), which have no ``.to_device`` of
    their own — ``_move_value`` handles Batch / tensor / list / dict / None
    uniformly."""
    if resp_track.segment is not None:
        resp_track.segment = resp_track.segment.to_device(device)
    resp_track.conditions = {k: _move_value(v, device) for k, v in resp_track.conditions.items()}
    if resp_track.advantages is not None:
        resp_track.advantages = resp_track.advantages.to(device=device)


class TrainStack(Remote):
    """Single-stage stage-driven train stack.

    Mirrors :class:`diffusionrl.training.stack.StageTrainStack` shape
    but for one stage only — no track-name dict, no optional-track
    semantics, no multi-track on_rollout_end fan-out.

    Created as a sibling ``Remote`` inside a placement block; takes
    handles to its FSDPBackend and StageAlgorithm siblings via
    sibling-handle auto-resolve.
    """

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        algorithm: StageAlgorithm,
        micro_batch_size: int,
        max_grad_norm: float,
    ) -> None:
        super().__init__()
        if int(micro_batch_size) < 1:
            raise ValueError(f"TrainStack.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"TrainStack.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.fsdp_backend = fsdp_backend
        self.algorithm = algorithm
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)

    def prepare_segment(self, resp_track: RolloutTrack) -> None:
        """Pre-step hook — call once per ``RolloutTrack`` before the
        ``num_updates_per_batch`` loop.  No-op if ``segment`` is None."""
        if resp_track.segment is None:
            return
        self.algorithm.prepare_segment(
            conditions=resp_track.conditions,
            segment=resp_track.segment,
        )

    def train(
        self,
        resp_track: RolloutTrack,
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Run one full optimizer step for this stage."""
        if resp_track.advantages is None:
            raise ValueError(
                "TrainStack.train: resp_track.advantages is None; "
                "upstream advantage pipeline must populate it before training."
            )

        bs = int(resp_track.batch_size)
        micro_slices = _build_micro_batch_slices(
            total_size=bs,
            micro_batch_size=int(self.micro_batch_size),
        )
        if not micro_slices:
            raise ValueError(f"TrainStack.train: empty batch (batch_size={bs}).")

        self.fsdp_backend.zero_grad()

        loss_scale = 1.0 / len(micro_slices)
        micros: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for start, end in micro_slices:
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            result = self.algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            micros.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated_metrics: Mapping[str, object] = aggregate_numeric_metrics([r.metrics for r in micros if r.metrics])

        if has_backward:
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning("TrainStack.train: no micro-batch reported backward; skipping optimizer step.")

        return TrainStepResult(
            loss=total_loss,
            grad_norm=grad_norm,
            lr=self._current_lr(),
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated_metrics,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    @distributed(dispatch_mode=Dispatch.DP_ALL)
    def train_track(
        self,
        resp_track: RolloutTrack,
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Driver-callable: prepare → train → on_rollout_end on the worker.

        Combines the three steps so worker-side mutations
        (``segment.sde_logp`` populated by ``prepare_segment``) flow into
        the subsequent ``train`` call without round-tripping through the
        driver. Dispatched ``DP_ALL`` so each DP worker receives its shard
        of ``resp_track``; per-shard loss/grad_norm/metrics merge back via
        ``pytree_merge``.
        """
        self._align_track_inputs(resp_track)
        self.prepare_segment(resp_track)
        result = self.train(resp_track, training_progress=float(training_progress))
        self.on_rollout_end()
        return result

    def _align_track_inputs(self, resp_track: RolloutTrack) -> None:
        """Move the track onto the model's device; see :func:`_align_track_to_model`."""
        device = next(self.fsdp_backend.trainable_module().parameters()).device
        _align_track_to_model(resp_track, device=device)

    def _current_lr(self) -> float:
        optimizer = self.fsdp_backend.optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                pass
        scheduler = self.fsdp_backend.scheduler
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            try:
                last = scheduler.get_last_lr()
                if isinstance(last, list) and last:
                    return float(last[0])
            except Exception:
                pass
        return 0.0


__all__ = [
    "TrainStack",
    "TrainStepResult",
]
