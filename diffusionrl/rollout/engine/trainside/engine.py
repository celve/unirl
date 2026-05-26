"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the trainable
stage, and exposes them as a :class:`BaseRolloutEngine`.  Used in
direct-sampling mode where the training model IS the sampler (on-policy
RL) and rollout runs in the same Python process as training — so no
worker subprocess and no weight sync are needed.
"""

from __future__ import annotations

from typing import Union

import torch

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
        stage: The trainable stage whose ``trainable_module()`` is the
            FSDP-wrapped model.  Used to scope eval/train mode around
            ``generate``.
    """

    _component_name = "trainside"

    def __init__(self, *, pipeline: Pipeline, stage: Stage) -> None:
        self.pipeline = pipeline
        self._model = stage.trainable_module()
        if hasattr(pipeline, "build_schedule_policy"):
            self.schedule_policy = pipeline.build_schedule_policy()
        else:
            self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                getattr(pipeline.bundle, "pretrained_path", None),
                shift=float(pipeline.shift),
            )

    def generate(self, req: RolloutReq) -> RolloutResp:
        ensure_req_sigmas(req, self.schedule_policy)
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                return self.pipeline.generate(req)
        finally:
            self._model.train(was_training)

    def shutdown(self) -> None:
        pass

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass

    def health_check(self) -> bool:
        return self.pipeline is not None and self._model is not None


__all__ = ["TrainsideRolloutEngine"]
