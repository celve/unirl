"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the
:class:`Policy` that holds its trainable module, and exposes them as a
:class:`BaseRolloutEngine`. Used in direct-sampling mode where the
training Policy IS the sampler (on-policy RL) and rollout runs in the
same Python process as training — so no worker subprocess and no
weight sync are needed.

Why this is just a ~50-line adapter:

- ``BaseRolloutEngine`` already specifies the ``generate(req) →
  RolloutResp`` contract. ``Pipeline.generate(req)`` returns exactly
  that shape, with conditions populated by the per-mode generate
  function (e.g. ``models/hunyuan_image3/modes/t2i.py:85``,
  ``models/sd3/pipeline.py:156``).
- The trainable module is FSDP-wrapped behind ``policy.model``; FSDP2
  auto-unshards on forward, including under ``eval()`` + ``no_grad()``.
  We only need to flip the module to eval mode for the rollout pass
  and restore the training-mode flag after.
- Memory lifecycle is owned by the ``FSDPPolicy`` lifecycle on the
  train actor, so ``sleep`` / ``wake_up`` are no-ops here.
- Weight sync is meaningless when sampler==trainer; the base class's
  ``NotImplementedError`` defaults are exactly right.
"""

from __future__ import annotations

import torch

from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from diffusionrl.training.policy import Policy
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler.

    Adapter that lets ``TrainActor`` participate in the
    :class:`RolloutPipelineMixin` host-contract by providing
    ``self.engine`` whose ``generate`` runs the materialized in-process
    pipeline under ``eval()`` + ``no_grad()``.

    Args:
        pipeline: A materialized ``models`` pipeline (e.g.
            ``HunyuanImage3Pipeline`` or ``SD3Pipeline``) whose
            ``generate(req)`` populates ``RolloutResp`` with
            conditions / rollout traces / decoded primitives.
        policy: The :class:`Policy` whose ``model`` attribute is the
            trainable module backing the pipeline's transformer
            (typically the outermost FSDP-wrapped policy). Used to
            scope eval/train mode around ``generate``.
    """

    _component_name = "trainside"

    def __init__(self, *, pipeline: Pipeline, policy: Policy) -> None:
        self.pipeline = pipeline
        self.policy = policy
        # Load the σ schedule policy from the pretrained checkpoint dir
        # the pipeline's bundle was built from. Dynamic-shift fields
        # come from scheduler.config.json — they're available here
        # because the trainside engine shares the process with the
        # bundle. (For sglang / vllm-omni engines the loader reads the
        # same JSONs directly without needing a Bundle.)
        # Per-Pipeline hook: ``build_schedule_policy`` lets a model declare
        # its dynamic-shift posture (Qwen-Image MUST be dynamic, etc.) and
        # provide canonical overrides for HF-repo-ID paths where
        # ``scheduler/scheduler_config.json`` can't be read locally.
        # Pipelines without the override get the default ``from_pretrained``
        # flow (static fallback for HF repo IDs, scheduler_config when local).
        if hasattr(pipeline, "build_schedule_policy"):
            self.schedule_policy = pipeline.build_schedule_policy()
        else:
            self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                getattr(pipeline.bundle, "pretrained_path", None),
                shift=float(pipeline.shift),
            )

    # ------------------------------------------------------------------
    # BaseRolloutEngine — generation
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run one rollout against the in-process pipeline.

        Wraps the call in ``eval()`` + ``no_grad()`` so the FSDP-wrapped
        transformer skips activation bookkeeping and dropout layers
        behave deterministically. Restores the prior training-mode
        flag on the policy's module before returning so a downstream
        ``train_optimizer_step`` call resumes training-mode forwards.

        Pins ``req.sigmas`` via :func:`ensure_req_sigmas` so the
        pipeline (and any replay path consuming
        ``segment.sigmas == req.sigmas``) sees the same schedule any
        other engine would have produced for the same request.
        """
        ensure_req_sigmas(req, self.schedule_policy)
        was_training = self.policy.model.training
        self.policy.eval()
        try:
            with torch.no_grad():
                return self.pipeline.generate(req)
        finally:
            self.policy.train(was_training)

    # ------------------------------------------------------------------
    # BaseRolloutEngine — lifecycle (memory owned by FSDPPolicy)
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """No worker subprocess to release; the pipeline outlives the engine."""

    def sleep(self) -> None:
        """No-op. Memory lifecycle is owned by the ``FSDPPolicy`` on the actor."""

    def wake_up(self) -> None:
        """No-op. See :meth:`sleep`."""

    def health_check(self) -> bool:
        return self.pipeline is not None and self.policy is not None


__all__ = ["TrainsideRolloutEngine"]
