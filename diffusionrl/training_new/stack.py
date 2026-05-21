"""Stage-driven training stack: dispatches per ``RolloutResp.rollout_traces`` slot.

A slot-keyed registry of :class:`StageAlgorithm` instances consumes a
``RolloutResp`` directly. The trainable-module facade is the
:class:`Policy` Protocol (``training_new/policy.py``). FSDP wrap, LoRA
injection and EMA shadow are all stackable policies composed via
:func:`compose_policy`. The stack itself holds a single ``policy``
reference; per-step it iterates the chain to locate optional surfaces
(``clip_grad_norm`` on FSDPPolicy, ``step`` on EMAPolicy) without
coupling to concrete classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import torch
from omegaconf import DictConfig

from diffusionrl.algorithms_new import AlgorithmStepResult, StageAlgorithm
from diffusionrl.training_new.policy import Policy, walk_source_chain
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


def _positive_int(*, name: str, value: Any) -> int:
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
class TrainOptimizerStepResult:
    """Result of one optimizer step under the stage-driven contract."""

    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    per_slot: Mapping[str, List[AlgorithmStepResult]]  # micro-step results, per slot
    metrics: Mapping[str, Any]


def _step_ema_in_chain(policy: Policy, step: Optional[int]) -> None:
    """Walk the Policy chain and call ``.step(step)`` on EVERY policy that
    implements it.

    Off-Protocol ``step`` is the EMA-update hook. Multiple policies may
    implement it independently (e.g. ``EMAPolicy.step`` for the eval
    shadow, ``NFTLoRAPolicy.step`` for the dual-adapter EMA when
    ``ema_update_timing == "optimizer_step"``). Detection via ``hasattr``
    keeps the Protocol surface narrow.
    """
    for p in walk_source_chain(policy):
        step_fn = getattr(p, "step", None)
        if callable(step_fn):
            step_fn(step)


def _on_rollout_end_in_chain(policy: Policy, step: Optional[int]) -> None:
    """Walk the Policy chain and call ``.on_rollout_end(step)`` on EVERY
    policy that implements it.

    Per-rollout-boundary hook used by policies whose update cadence is
    keyed to rollout cycles rather than optimizer steps — e.g.
    :class:`NFTLoRAPolicy` with ``ema_update_timing == "rollout_end"``.
    No-op for policies that don't expose the method (most policies).
    """
    for p in walk_source_chain(policy):
        fn = getattr(p, "on_rollout_end", None)
        if callable(fn):
            fn(step)


@dataclass
class StageTrainStack:
    """Stage-driven train stack: dispatches per ``RolloutResp.rollout_traces`` slot.

    A slot-keyed registry of :class:`StageAlgorithm` instances consumes a
    ``RolloutResp`` directly.

    The trainable-module facade is a single :class:`Policy` (typically the
    outermost layer of a ``compose_policy(stage, [lora_cfg, fsdp_cfg,
    ema_cfg])`` stack). The stack relies on the Policy for grad clipping
    (``policy.clip_grad_norm`` when present) and discovers an optional EMA
    shadow by walking the source chain.
    """

    policy: Policy
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    algorithms: Dict[str, StageAlgorithm]
    cfg: DictConfig
    # Slot names registered in ``algorithms`` that are allowed to be
    # absent from ``resp.rollout_traces`` without raising. Default:
    # empty — every registered algorithm MUST see its slot in the
    # response, otherwise the missing slot is a silent no-train risk
    # (e.g. multi-modal RL where a rollout actor fails to emit the AR
    # trace and the AR head silently never updates while training "looks
    # normal"). Per-task topologies (e.g. an SD3-only run that registers
    # an AR algorithm but never runs AR rollouts) opt in by passing the
    # missing slot here: ``optional_slots=frozenset({"ar"})``.
    optional_slots: FrozenSet[str] = field(default_factory=frozenset)
    _optimizer_step: int = field(default=0, init=False, repr=False)

    def train_microbatch(
        self,
        resp: RolloutResp,
        *,
        training_progress: float,
        loss_scale: float,
    ) -> Dict[str, AlgorithmStepResult]:
        """Run one forward + scaled backward across all configured slots.

        Slots in ``self.algorithms`` that are absent from
        ``resp.rollout_traces`` raise ``ValueError`` by default — silent
        skipping is a multi-modal-RL silent-no-train risk. Per-task
        topologies that genuinely want a registered algorithm to remain
        idle when its slot is absent opt in via ``optional_slots``
        (e.g. ``optional_slots=frozenset({"ar"})`` for an SD3-only run
        that still has an AR algorithm wired but skips it when no AR
        trace is emitted).

        Does not touch the optimizer; callers manage the
        ``zero_grad`` / ``step`` cadence (typically :meth:`train_optimizer_step`).
        """
        if resp.advantages is None:
            raise ValueError(
                "StageTrainStack.train_microbatch requires resp.advantages "
                "to be populated upstream (rollout / advantage pipeline)."
            )
        results: Dict[str, AlgorithmStepResult] = {}
        for slot, alg in self.algorithms.items():
            seg = resp.rollout_traces.get(slot)
            if seg is None:
                if slot in self.optional_slots:
                    continue
                raise ValueError(
                    f"StageTrainStack.train_microbatch: slot {slot!r} is "
                    f"registered in ``algorithms`` but absent from "
                    f"``resp.rollout_traces`` (keys: "
                    f"{sorted(resp.rollout_traces.keys())}). Silent skipping "
                    f"would let a missing trace appear as 'no training step' "
                    f"for that head (silent no-train). Either ensure the "
                    f"rollout actor emits the {slot!r} trace, or opt in to "
                    f"silent skipping by passing "
                    f"``optional_slots=frozenset({{...{slot!r}...}})`` to "
                    f"``StageTrainStack``."
                )
            results[slot] = alg.compute_loss_and_backward(
                conditions=resp.conditions,
                segment=seg,
                advantages=resp.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
        return results

    def train_optimizer_step(
        self,
        resp: RolloutResp,
        *,
        training_progress: float,
    ) -> TrainOptimizerStepResult:
        """One optimizer step over ``resp``.

        Slices the response into micro-batches per
        ``cfg.training.plan.micro_batch_size``, accumulates scaled backward
        across micros, then clips + steps + EMA once at the end.
        """
        self.optimizer.zero_grad()

        bs = int(resp.batch_size)
        micro_batch_size = int(self.cfg.training.plan.micro_batch_size)
        micro_slices = _build_micro_batch_slices(
            total_size=bs,
            micro_batch_size=micro_batch_size,
        )
        if not micro_slices:
            raise ValueError("train_optimizer_step requires a non-empty batch.")

        loss_scale = 1.0 / len(micro_slices)
        per_slot_micros: Dict[str, List[AlgorithmStepResult]] = {slot: [] for slot in self.algorithms}
        total_loss = 0.0
        has_backward = False

        # Fast-path: with a single micro-batch covering the whole optimizer-step
        # batch, skip ``resp.slice``. The slice is otherwise a no-op semantically
        # but materially mutates condition tensors that carry CFG-doubled
        # batch dims (e.g. ``FusedMultimodalCondition.input_ids`` is CONCAT
        # but ``rope_cache`` is SHARED — sample-level slicing breaks the
        # internal CFG layout). Avoiding the slice when bs == micro is
        # both cheaper and preserves the upstream condition shape.
        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for start, end in micro_slices:
            micro_resp = resp if single_micro else resp.slice(start, end)
            slot_results = self.train_microbatch(
                micro_resp,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            for slot, result in slot_results.items():
                per_slot_micros.setdefault(slot, []).append(result)
                total_loss += result.loss
                has_backward = has_backward or result.has_backward

        # Aggregate metrics across micros AND across slots; per-slot metric
        # keys are namespaced under ``<slot>/<key>`` so wandb stays readable.
        namespaced_metric_dicts: List[Mapping[str, Any]] = []
        for slot, micros in per_slot_micros.items():
            for r in micros:
                if r.metrics:
                    namespaced_metric_dicts.append({f"{slot}/{k}": v for k, v in r.metrics.items()})
        aggregated_metrics: Dict[str, Any] = aggregate_numeric_metrics(namespaced_metric_dicts)

        if has_backward:
            max_grad_norm = float(self.cfg.training.execution.max_grad_norm)
            # Walk the Policy chain to find ``clip_grad_norm``. The outermost
            # policy (e.g. ``EMAPolicy``) typically does NOT expose grad
            # clipping; ``FSDPPolicy`` does — and its DTensor-aware
            # implementation is required when params are FSDP-sharded.
            clip_fn = next(
                (fn for p in walk_source_chain(self.policy) if callable(fn := getattr(p, "clip_grad_norm", None))),
                None,
            )
            if clip_fn is not None:
                clipped = clip_fn(max_grad_norm)
            else:
                # No-FSDP fallback (unit-test path): plain torch clip on the
                # Policy's exposed parameters. FSDPPolicy provides its own
                # DTensor-aware ``clip_grad_norm`` for production.
                clipped = torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()), max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            _step_ema_in_chain(self.policy, self._optimizer_step)
            self._optimizer_step += 1
        else:
            clipped = 0.0
            logger.warning(
                "StageTrainStack.train_optimizer_step: no slot reported backward; skipping optimizer step",
            )

        if clipped is None:
            grad_norm_value = 0.0
        elif isinstance(clipped, torch.Tensor):
            grad_norm_value = float(clipped.item())
        else:
            grad_norm_value = float(clipped)

        return TrainOptimizerStepResult(
            loss=total_loss,
            grad_norm=grad_norm_value,
            lr=self._current_lr(),
            has_backward=has_backward,
            per_slot=per_slot_micros,
            metrics=aggregated_metrics,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — dispatches to any Policy in the
        chain that implements ``on_rollout_end(step)``.

        Called by ``new_train_actor._train_resp`` after a rollout's
        ``train_optimizer_step`` has finished. Used by
        :class:`NFTLoRAPolicy` with ``ema_update_timing == "rollout_end"``
        to advance its dual-adapter EMA exactly once per rollout cycle
        (independent of the optimizer-step count inside the rollout).
        """
        _on_rollout_end_in_chain(self.policy, self._optimizer_step)

    def _current_lr(self) -> float:
        """Best-effort LR lookup; same heuristic as the legacy ``TrainStack``."""
        param_groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                pass
        if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
            try:
                last = self.scheduler.get_last_lr()
                if isinstance(last, list) and last:
                    return float(last[0])
            except Exception:
                pass
        return 0.0


__all__ = ["TrainOptimizerStepResult", "StageTrainStack"]
