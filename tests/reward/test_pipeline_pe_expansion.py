"""Unit tests for the PE-joint text-expansion assert in
:meth:`RewardPipeline.score_and_attach`.

The legacy heuristic only checked ``len(sample_ids) % len(texts) == 0`` — an
accidentally divisible factor (e.g. resp=2× when N×M=4) would silently
mis-replicate. The tightened assert cross-checks the factor against
``stage_params['ar']['n'] * stage_params['diffusion']['num_samples_per_prompt']``.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutTrack
from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
)
from diffusionrl.types.segments.latent import LatentSegment


class _StubRewardService:
    """Minimal RewardService stub — the assert under test fires BEFORE any
    executor is touched. ``preferred_input_kind = 'image'`` is enough."""

    preferred_input_kind = "image"


def _build_pipeline() -> RewardPipeline:
    return RewardPipeline(_StubRewardService())  # type: ignore[arg-type]


def _build_req(*, texts: List[str], n: int, m: int) -> RolloutReq:
    p = len(texts)
    sample_ids = [f"p{i}" for i in range(p)]
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=sample_ids,
        primitives={"text": Texts(texts=list(texts))},
        request_conditions={},
        sampling_params=ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(samples_per_prompt=m),
            ar=ARSamplingParams(samples_per_prompt=n),
        ),
    )


def _build_image_track(*, batch: int) -> RolloutTrack:
    return RolloutTrack(
        sample_ids=[f"s{i}" for i in range(batch)],
        parent_ids=None,
        parent_track=None,
        segment=LatentSegment(
            sample_indices=torch.arange(batch, dtype=torch.long),
            positions=torch.zeros(batch, dtype=torch.long),
            latents=torch.zeros(batch, 2, 4, 4, 4),
            sigmas=torch.linspace(1.0, 0.0, 3),
            indices=torch.arange(2, dtype=torch.long),
        ),
        decoded=Images.from_tensor(torch.zeros(batch, 3, 4, 4)),
    )


def test_score_and_attach_raises_when_factor_disagrees_with_stage_params():
    """Mismatched implicit factor vs explicit N*M must raise — previously silent."""
    # texts=2 (one prompt × ar.n=1 explicit but track has 4× samples — implicit
    # factor=2 disagrees with explicit N*M=1*1=1).
    pipeline = _build_pipeline()
    req = _build_req(texts=["a", "b"], n=1, m=1)
    track = _build_image_track(batch=4)  # implicit factor 4/2 = 2; expected 1*1 = 1
    with pytest.raises(RuntimeError, match=r"does not match sampling_params N\*M"):
        pipeline.score_and_attach(req=req, track=track)


def test_score_and_attach_accepts_matching_factor():
    """Matching implicit factor must NOT raise on the factor-check path."""
    pipeline = _build_pipeline()
    # texts=2, sample_ids=8 → implicit factor 4; explicit N*M = 2*2 = 4 → OK
    # We don't drive the reward service through, but the factor check happens
    # first; we expect to advance past it and fail later on the stub service.
    req = _build_req(texts=["a", "b"], n=2, m=2)
    track = _build_image_track(batch=8)
    # The stub service doesn't implement compute(); whatever comes after the
    # factor check raises an AttributeError or similar — but specifically NOT
    # our factor-mismatch RuntimeError.
    with pytest.raises(Exception) as ei:
        pipeline.score_and_attach(req=req, track=track)
    assert "does not match sampling_params" not in str(ei.value)
