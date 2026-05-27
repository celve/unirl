"""Verify the tightened shard-factor assert in
:meth:`RolloutPipelineMixin.generate_buffered`.

Replaces the legacy ``% == 0`` heuristic — which silently mis-paired
shards on any accidentally divisible mismatch — with an explicit
cross-check against ``stage_params['ar']['n'] *
stage_params['diffusion']['num_samples_per_prompt']``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytest
import torch

# Warm import graph past pre-existing circular-import seam in diffusionrl.distributed.
import diffusionrl.config  # noqa: F401
from diffusionrl.ray.mixins.rollout_pipeline import RolloutPipelineMixin
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
)
from diffusionrl.types.segments.latent import LatentSegment


@dataclass
class _FakeHandle:
    id: str


def _make_resp_with_k_groups(*, k: int, samples_per_group: int = 1) -> RolloutResp:
    """Build a single-track RolloutResp whose ``split()`` returns ``k`` shards."""
    total = k * samples_per_group
    sample_ids = [f"s{i}" for i in range(total)]
    group_ids = [f"g{i // samples_per_group}" for i in range(total)]
    track = RolloutTrack(
        sample_ids=sample_ids,
        parent_ids=group_ids,
        parent_track=None,
        segment=LatentSegment(
            sample_indices=torch.arange(total, dtype=torch.long),
            positions=torch.zeros(total, dtype=torch.long),
            latents=torch.zeros(total, 2, 4, 4, 4),
            sigmas=torch.linspace(1.0, 0.0, 3),
            indices=torch.arange(2, dtype=torch.long),
        ),
    )
    return RolloutResp(tracks={"image": track})


def _make_req_with_p_groups(*, p: int, n: int, m: int) -> RolloutReq:
    sample_ids = [f"p{i}" for i in range(p)]
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=sample_ids,
        primitives={"text": Texts(texts=[f"text_{i}" for i in range(p)])},
        request_conditions={},
        sampling_params=ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(samples_per_prompt=m),
            ar=ARSamplingParams(samples_per_prompt=n),
        ),
    )


class _FactorTestHost(RolloutPipelineMixin):
    """Minimal host: stubs ``generate``/``put_buffer`` and inherits
    ``_split_req_by_group`` from the mixin."""

    def __init__(self, resp: RolloutResp) -> None:
        self._resp = resp
        self._buf: Dict[str, RolloutResp] = {}

    def generate(self, req: RolloutReq) -> RolloutResp:
        return self._resp

    def put_buffer(self, meta, resp):  # noqa: ANN001 — only called on the success path
        handle = _FakeHandle(id=f"h{len(self._buf)}")
        self._buf[handle.id] = resp
        return handle


def test_generate_buffered_raises_when_factor_disagrees_with_stage_params():
    """Implicit shard factor must equal explicit N*M from stage_params."""
    # 4 resp groups vs 2 req groups → implicit factor 2; expected N*M = 1*1 = 1.
    resp = _make_resp_with_k_groups(k=4)
    req = _make_req_with_p_groups(p=2, n=1, m=1)
    host = _FactorTestHost(resp)
    with pytest.raises(RuntimeError, match=r"does not match sampling_params N\*M"):
        host.generate_buffered(req)


def test_generate_buffered_accepts_matching_factor():
    """4 resp groups vs 2 req groups with N=2, M=1 (or any N*M=2) → factor matches."""
    resp = _make_resp_with_k_groups(k=4)
    req = _make_req_with_p_groups(p=2, n=2, m=1)
    host = _FactorTestHost(resp)
    handles = host.generate_buffered(req)
    # 4 resp shards × 1 buffered per shard → 4 handles.
    assert len(handles) == 4
