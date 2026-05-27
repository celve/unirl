"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the trainable
stage, and exposes them as a :class:`BaseRolloutEngine`.  Used in
direct-sampling mode where the training model IS the sampler (on-policy
RL) and rollout runs in the same Python process as training — so no
worker subprocess and no weight sync are needed.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.models.types.ar import ARStage
from diffusionrl.models.types.diffusion import DiffusionStage
from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

Stage = Union[DiffusionStage, ARStage]


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler.

    Args:
        pipeline: A materialized ``models`` pipeline whose
            ``generate(req)`` populates ``RolloutResp``.
        stage: Optional trainable stage whose ``trainable_module()`` is
            the FSDP-wrapped model. Used to scope eval/train mode around
            ``generate``. When omitted, resolved from
            ``pipeline.<stage_attr>`` (default ``"diffusion"``).
        stage_attr: Attribute on ``pipeline`` to read the stage from when
            ``stage`` is not explicitly provided.
        forward_batch_size: Optional intra-call chunk size for the
            ``pipeline.generate`` forward path. When set and the request
            exceeds this, ``generate`` slices the request via
            :meth:`RolloutReq.slice`, runs ``pipeline.generate`` per chunk,
            and concatenates results via :meth:`RolloutResp.concat`.
            Mirrors :attr:`diffusionrl.rollout.plan.RolloutPlan.forward_batch_size`,
            which the v1 driver feeds to
            :func:`diffusionrl.rollout.engine.chunked_engine_generate_req`
            for the same purpose — bounding stage peak memory (e.g. SD3
            VAE decode) when there is no external inference runtime to
            chunk for us.
    """

    _component_name = "trainside"

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage: Optional[Stage] = None,
        stage_attr: str = "diffusion",
        forward_batch_size: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self._model = stage.trainable_module()
        if forward_batch_size is not None and forward_batch_size < 1:
            raise ValueError(
                f"TrainsideRolloutEngine.forward_batch_size must be >= 1 when set; got {forward_batch_size!r}"
            )
        self.forward_batch_size = forward_batch_size
        if isinstance(stage, DiffusionStage):
            if hasattr(pipeline, "build_schedule_policy"):
                self.schedule_policy = pipeline.build_schedule_policy()
            else:
                self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                    getattr(pipeline.bundle, "pretrained_path", None),
                    shift=float(pipeline.shift),
                )
        else:
            # AR stage — no diffusion schedule needed
            self.schedule_policy = None

    @distributed(dispatch_mode=Dispatch.DP_ALL)
    def generate(self, req: RolloutReq) -> RolloutResp:
        if self.schedule_policy is not None:
            ensure_req_sigmas(req, self.schedule_policy)
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                fbs = self.forward_batch_size
                bs = int(req.batch_size)
                if fbs is None or bs <= fbs:
                    return self.pipeline.generate(req)
                outputs: List[RolloutResp] = []
                for start in range(0, bs, fbs):
                    end = min(start + fbs, bs)
                    outputs.append(self.pipeline.generate(req.slice(start, end)))
                    torch.cuda.empty_cache()
                return RolloutResp.concat(outputs)
        finally:
            self._model.train(was_training)

    def shutdown(self) -> None:
        pass

    # sleep / wake_up inherit BaseRolloutEngine's @distributed no-op default.

    def health_check(self) -> bool:
        return self.pipeline is not None and self._model is not None


__all__ = ["TrainsideRolloutEngine"]
