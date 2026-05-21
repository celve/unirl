"""Stage-driven GRPO: ``DiffusionGRPO`` over a ``LatentSegment`` and
``ARGRPO`` over a ``TextSegment``.

Both implement :class:`StageAlgorithm` and share the module-level
``_grpo_clip_loss`` / ``_resolve_clip_range_from_schedule`` helpers so their
loss math stays identical. CFG batching, predict_noise, SDE math, autocast,
and per-step iteration are owned by ``stage.replay(...)``; the algorithm is
~20 lines of ratio-clip math.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from diffusionrl.types.conditions import Condition
from diffusionrl.types.segments.latent import LatentSegment
from diffusionrl.types.segments.text import TextSegment

from .base import AlgorithmStepResult, StageAlgorithm


def _resolve_clip_range_from_schedule(clip_range: float, schedule: str, progress: float) -> float:
    """Schedule-aware clip range. Mirrors ``GRPOAlgorithm.get_clip_range``."""
    if schedule == "linear_decay":
        return clip_range * (1.0 - 0.5 * float(progress))
    if schedule == "cosine_decay":
        return clip_range * (0.5 * (1.0 + math.cos(math.pi * float(progress))))
    return clip_range


def _grpo_clip_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-style clipped objective. Element-wise; reduction is the caller's job.

    All inputs must be broadcastable to a common shape. Returns
    ``(loss_per_element, ratio_metrics_dict)``. The metrics tensors are
    detached scalars suitable for logging.
    """
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss_per_elem = torch.maximum(unclipped, clipped)

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_range).float().mean().detach(),
        "clipfrac_gt_one": (ratio - 1.0 > clip_range).float().mean().detach(),
        "clipfrac_lt_one": (1.0 - ratio > clip_range).float().mean().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
    }
    return loss_per_elem, metrics


def _typed_conditions(
    conditions: Mapping[str, Condition],
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


class DiffusionGRPO(StageAlgorithm):
    """GRPO over a diffusion ``LatentSegment`` via ``DiffusionStage.replay``.

    The whole forward path (CFG batching, noise prediction, SDE math, autocast,
    per-step iteration) is owned by :meth:`DiffusionStage.replay`; this class
    is pure ratio-clip math against ``segment.sde_logp``.

    Args:
        stage: The :class:`DiffusionStage` whose ``replay`` produces new
            log-probs aligned with ``segment.sde_logp[:, slot_for_steps]``.
        params: The per-call params object the stage's ``replay`` consumes
            (e.g. ``SD3DiffusionParams``). Held as algorithm state so the
            dispatcher doesn't need to know it.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"`` — applied via ``training_progress``.
        conditions_cls: Stage-typed conditions container with a
            ``from_dict(Mapping[str, Condition])`` classmethod. ``None``
            forwards the dict verbatim (unit-test path).
    """

    def __init__(
        self,
        *,
        stage: Any,
        params: Any,
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        self.stage = stage
        self.params = params
        self.clip_range = float(clip_range)
        self.clip_schedule = str(clip_schedule)
        self.conditions_cls = conditions_cls

    def replay_old_log_probs(self, conditions: Mapping[str, "Condition"], segment: "LatentSegment") -> None:
        """Pre-compute old_logp for replay mode and store on segment.

        No-op if segment.sde_logp is already populated (native mode).
        Called once before multi-update training so old_logp is frozen at
        the pre-update weights.
        """
        if segment.sde_logp is not None or segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        typed_conds = _typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        segment.sde_logp = result.log_probs.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = _typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs  # [B, S']

        old_logp = self._gather_old_sde_logp(segment, target_steps).to(dtype=new_logp.dtype, device=new_logp.device)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )
        loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    # -- helpers --------------------------------------------------------

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment.

        Subclasses can override to apply skip-last / skip-initial filtering or
        to honor a training-indices schedule; the default trains every step
        the rollout recorded.
        """
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]

    @staticmethod
    def _gather_old_sde_logp(segment: "LatentSegment", target_steps: List[int]) -> torch.Tensor:
        """Slice ``segment.sde_logp[:, slot_for_each_target_step]`` → ``[B, S']``."""
        if segment.sde_logp is None or segment.sde_indices is None:
            raise ValueError(
                "DiffusionGRPO requires segment.sde_logp and segment.sde_indices "
                "(rollout must record SDE log-probs for the steps you train on)."
            )
        sde_indices_list = [int(i) for i in segment.sde_indices.tolist()]
        try:
            positions = [sde_indices_list.index(i) for i in target_steps]
        except ValueError as exc:
            raise ValueError(
                f"DiffusionGRPO target step not present in segment.sde_indices: "
                f"target_steps={target_steps}, sde_indices={sde_indices_list}"
            ) from exc
        return segment.sde_logp[:, positions]


class ARGRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``.

    The teacher-forced forward and per-token log-prob recompute is owned by
    :meth:`ARStage.replay`; this class expands per-sample advantages to per-
    token via ``cu_seqlens`` and runs the same PPO clip math.

    Args:
        stage: The :class:`ARStage` whose ``replay`` produces packed-varlen
            new log-probs aligned with ``segment.log_probs``.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"``.
        conditions_cls: Stage-typed conditions container with
            ``from_dict(Mapping[str, Condition])``.
    """

    def __init__(
        self,
        *,
        stage: Any,
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_schedule = str(clip_schedule)
        self.conditions_cls = conditions_cls

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = _typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment)  # [total_tokens]
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
        )
        loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``.

        Each sample's advantage is repeated across its ``lengths``-defined
        token span so that token positions in segment ``k`` all see
        ``advantages[k]``. ``lengths`` comes from
        :attr:`Batched.lengths` on the segment (derived from the framework-
        managed cu_seqlens).
        """
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"ARGRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["ARGRPO", "DiffusionGRPO"]
