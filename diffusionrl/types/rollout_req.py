"""RolloutReq — top-level SoA container for one rollout's worth of inputs.

Pairs with ``RolloutResp`` (in ``diffusionrl/types/rollout_resp.py``). Carries:

- ``primitives: Dict[str, Texts | Images | Videos | Audios]`` — raw inputs
  keyed by modality-slot name (``"text"``, ``"image"``, ...). The pipeline
  encodes each via the relevant ``EncodeStage`` / ``EmbedStage`` before
  generation.
- ``request_conditions: Dict[str, Condition]`` — precomputed encoded inputs
  the engine should consume verbatim instead of (re-)deriving from
  ``primitives``. Symmetric with ``RolloutResp.conditions``. Key convention:
  ``"initial_latents"`` → ``ImageLatentCondition(latents=x_T_or_init_img)``
  for engines that accept a precomputed start-of-denoising tensor (SGLang's
  ``Req.latents``, vllm-omni's per-stage init-latents). Future keys: other
  typed engine-bound inputs land under their own slot.
- ``stage_params: Dict[str, Any]`` — bag-of-options keyed by stage name
  (``"diffusion"`` → eta, sde_indices, num_inference_steps, height, width,
  guidance_scale, ...; ``"ar"`` → ``ARSamplingParams``, etc.).
  Architecture's stages read their key.
- ``sigmas: Optional[torch.Tensor]`` — the σ schedule for this rollout,
  computed main-side via
  :func:`diffusionrl.sde.runtime.ensure_req_sigmas` (which applies the
  engine's :class:`FlowMatchSchedulePolicy` to the per-request
  ``(T, H, W)`` triple) and populated by the rollout-engine adapter
  just before dispatch. This is
  the **single source of truth** for σ across all rollout backends
  (trainside / sglang / vllm-omni): every engine MUST consume this
  schedule rather than computing its own, and the response handler
  asserts the schedule the engine actually used (echoed back via
  ``LatentSegment.sigmas``) matches what was sent. ``None`` only at
  request-construction time (driver-side ``plan_requests``); engines
  populate it before forwarding. Shape ``[T+1]`` (length includes the
  terminal 0), values in ``[0, 1]``, ``float32``, host-device-agnostic
  (engines move to worker device when serializing).
- ``sample_ids`` / ``group_ids`` — mirror ``RolloutResp`` so request and
  response can be correlated by ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.conditions.base import Condition
from diffusionrl.types.primitives import Audios, Images, Texts, Videos
from diffusionrl.utils.batched import FieldKind, concat_field, field, shared_field

PrimitiveValue = Union[Texts, Images, Videos, Audios]


@dataclass
class RolloutReq(Transportable):
    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)
    primitives: Dict[str, PrimitiveValue] = field(kind=FieldKind.CONCAT, transport=True, default_factory=dict)
    request_conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, transport=True, default_factory=dict)
    stage_params: Dict[str, Any] = shared_field(default_factory=dict)
    # σ schedule is shared across all samples in the request — every
    # sample runs the same num_inference_steps / shift / dynamic-shift μ
    # by construction (geometry varies per-sample only via height/width,
    # but plan_requests / driver fix those per-batch). Hence ``shared_field``.
    sigmas: Optional[torch.Tensor] = shared_field(default=None)

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size


__all__ = ["RolloutReq", "PrimitiveValue"]
