"""Per-track composed train stack.

Holds N parallel :class:`TrainTrack` instances keyed by track name
(matching ``RolloutResp.tracks`` keys).  Each track runs an independent
``zero_grad -> micro-loop -> clip -> step -> scheduler.step -> EMA.step``
cycle via :meth:`StageTrainStack.train_track`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Tuple

import torch

from diffusionrl.algorithms import AlgorithmStepResult
from diffusionrl.training.fsdp_utils import clip_grad_norm, trainable_params
from diffusionrl.training.train_track import TrainTrack
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class TrackMiniBatchResult:
    """Result of one full optimizer step on one track."""

    track_name: str
    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    micros: List[AlgorithmStepResult]
    metrics: Mapping[str, object]


@dataclass
class StageTrainStack:
    """Multi-track stage-driven train stack.

    Holds :class:`TrainTrack` instances keyed by track name.  Each track
    runs an independent training cycle via :meth:`train_track`.  Direct
    calls to PyTorch, the algorithm, the optimizer, and EMA — no
    ``walk_source_chain``, no ``hasattr``.
    """

    tracks: Dict[str, TrainTrack]
    max_grad_norm: float
    optional_tracks: FrozenSet[str] = field(default_factory=frozenset)
    _optimizer_steps: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        track_keys = set(self.tracks)
        bad_optional = self.optional_tracks - track_keys
        if bad_optional:
            raise ValueError(
                f"optional_tracks {sorted(bad_optional)} are not in tracks; "
                f"optional_tracks must be a subset of track keys {sorted(track_keys)}."
            )
        for name, track in self.tracks.items():
            if int(track.micro_batch_size) < 1:
                raise ValueError(f"tracks[{name!r}].micro_batch_size must be >= 1; got {track.micro_batch_size}.")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0; got {self.max_grad_norm}.")
        self._optimizer_steps = {name: 0 for name in track_keys}

    def train_track(
        self,
        resp: RolloutResp,
        track_name: str,
        *,
        training_progress: float,
    ) -> TrackMiniBatchResult:
        """Run one full optimizer step for the named track.

        When ``track.algorithms`` holds multiple entries (HI3 shared-
        backbone case), each algorithm's gradients are accumulated under
        a single ``zero_grad`` / ``step`` pair — replicating the pre-#136
        ``StageTrainStack.train_microbatch`` "many algorithms, one
        optimizer step" semantic.
        """
        if track_name not in self.tracks:
            raise ValueError(
                f"StageTrainStack.train_track: track {track_name!r} "
                f"is not registered. Known tracks: {sorted(self.tracks)}."
            )
        track = self.tracks[track_name]

        # Collect resp.tracks slot for each algorithm hosted by this track.
        resp_slots: Dict[str, object] = {}
        for alg_key in track.algorithms:
            slot = resp.tracks.get(alg_key)
            if slot is None:
                if track_name in self.optional_tracks:
                    continue
                raise ValueError(
                    f"StageTrainStack.train_track: track {track_name!r} "
                    f"hosts algorithm slot {alg_key!r} but resp.tracks has "
                    f"no entry for that slot (resp keys: "
                    f"{sorted(resp.tracks.keys())}). Either ensure the "
                    f"rollout pipeline emits the {alg_key!r} track, or opt "
                    f"in to silent skipping via optional_tracks."
                )
            if slot.advantages is None:
                raise ValueError(
                    f"StageTrainStack.train_track: track {track_name!r} "
                    f"algorithm slot {alg_key!r} has advantages=None; the "
                    f"upstream advantage pipeline must populate "
                    f"resp.tracks[{alg_key!r}].advantages."
                )
            resp_slots[alg_key] = slot

        if not resp_slots:
            return TrackMiniBatchResult(
                track_name=track_name,
                loss=0.0,
                grad_norm=0.0,
                lr=self._current_lr(track_name),
                has_backward=False,
                micros=[],
                metrics={},
            )

        model = track.stage.trainable_module()

        track.optimizer.zero_grad()

        # All algorithm slots in this track share the same batch_size by
        # construction (rollout pipeline emits 1:1 sample lineage).
        bs = int(next(iter(resp_slots.values())).batch_size)
        micro_batch_size = int(track.micro_batch_size)
        micro_slices = _build_micro_batch_slices(
            total_size=bs,
            micro_batch_size=micro_batch_size,
        )
        if not micro_slices:
            raise ValueError(f"train_track({track_name!r}): track has empty batch (batch_size={bs}).")

        loss_scale = 1.0 / len(micro_slices)
        micros: List[AlgorithmStepResult] = []
        per_alg_results: Dict[str, List[AlgorithmStepResult]] = {k: [] for k in resp_slots}
        total_loss = 0.0
        has_backward = False
        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)

        for start, end in micro_slices:
            for alg_key, slot in resp_slots.items():
                micro = slot if single_micro else slot.slice(start, end)
                result = track.algorithms[alg_key].compute_loss_and_backward(
                    conditions=micro.conditions,
                    segment=micro.segment,
                    advantages=micro.advantages,
                    training_progress=training_progress,
                    loss_scale=loss_scale,
                )
                micros.append(result)
                per_alg_results[alg_key].append(result)
                total_loss += result.loss
                has_backward = has_backward or result.has_backward

        # Aggregate metrics per algorithm. Multi-algo case prefixes the slot
        # name (image/policy_loss vs ar/policy_loss) to avoid key collision.
        aggregated_metrics: Dict[str, object] = {}
        multi_alg = len(per_alg_results) > 1
        for alg_key, results in per_alg_results.items():
            metrics = aggregate_numeric_metrics([r.metrics for r in results if r.metrics])
            for k, v in metrics.items():
                aggregated_metrics[f"{alg_key}/{k}" if multi_alg else k] = v

        if has_backward:
            params = list(trainable_params(model))
            clipped = clip_grad_norm(params, self.max_grad_norm)
            track.optimizer.step()
            if track.scheduler is not None:
                track.scheduler.step()
            if track.ema is not None:
                track.ema.step(self._optimizer_steps[track_name])
            self._optimizer_steps[track_name] += 1
        else:
            clipped = 0.0
            logger.warning(
                "StageTrainStack.train_track[%s]: no micro-batch reported backward; skipping optimizer step.",
                track_name,
            )

        if clipped is None:
            grad_norm_value = 0.0
        elif isinstance(clipped, torch.Tensor):
            grad_norm_value = float(clipped.item())
        else:
            grad_norm_value = float(clipped)

        return TrackMiniBatchResult(
            track_name=track_name,
            loss=total_loss,
            grad_norm=grad_norm_value,
            lr=self._current_lr(track_name),
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated_metrics,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — dispatches per-track."""
        for name, track in self.tracks.items():
            if track.ema is not None:
                track.ema.on_rollout_end(self._optimizer_steps[name])

    def _current_lr(self, track_name: str) -> float:
        track = self.tracks[track_name]
        param_groups = getattr(track.optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                pass
        if track.scheduler is not None and hasattr(track.scheduler, "get_last_lr"):
            try:
                last = track.scheduler.get_last_lr()
                if isinstance(last, list) and last:
                    return float(last[0])
            except Exception:
                pass
        return 0.0


__all__ = [
    "StageTrainStack",
    "TrackMiniBatchResult",
]
