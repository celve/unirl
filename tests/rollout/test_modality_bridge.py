"""Layer 2 bridge tests — modality-agnostic dispatch + video noise route.

Scope is intentionally narrow: this file covers only the bridge points
whose verification does NOT require importing
``diffusionrl.types.sample`` or ``diffusionrl.types.response`` /
``diffusionrl.types.rollout_resp``, because those modules trip a
pre-existing circular-import bug under pytest collection. Specifically
tests here verify:

- ``_build_diffusion_stage_params``: passthrough of ``num_frames`` /
  ``init_same_noise`` (+ image-path regression).
- ``compute_initial_noise_for_request``: video latent-shape route for
  ``wan21_t2v`` / ``wan22_t2v`` (+ SD3 regression + unsupported modality
  fallback).

Verification of the other Layer 2 bridges
-----------------------------------------

The remaining bridges (``resp_to_samples`` modality dispatch,
``MediaPreview.videos`` tri-state validation, ``attach_media_preview``
video path, ``log_generated_media`` ``wandb.Video`` output) are
verifiable by code review against the implementations in:

- ``diffusionrl/rollout/engine/types_compat.py::resp_to_samples``
- ``diffusionrl/types/sample.py::MediaPreview``
- ``diffusionrl/types/response.py::attach_media_preview``
- ``diffusionrl/utils/wandb_logger.py::log_generated_media``

Adding pytest coverage for those will land in a follow-up that first
breaks the circular dependency between ``types.sample`` and
``types.rollout_resp`` (e.g. via a ``TYPE_CHECKING`` guard on the
``rollout_resp -> sample.MediaPreview`` import or a lazy MediaPreview
import inside ``rollout.engine.base``).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

# Warm import graph past pre-existing circular imports in
# ``diffusionrl.distributed`` → ``rollout.engine`` → ``types.rollout_req``.
# Same warm pattern as ``tests/test_primitives_packed.py``.
import diffusionrl.config  # noqa: F401  -- import-graph warm
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.sampling import SamplingParams, SDEConfig

# Modality / LatentSegment imports are deferred to test bodies — touching
# ``diffusionrl.types.segments`` or ``diffusionrl.types.conditions`` at
# module import time triggers an unrelated pre-existing circular import
# (``distributed`` → ``rollout.engine`` → ``types.rollout_resp`` → back to
# ``segments``/``conditions``). Importing inside test bodies defers past
# pytest collection so the cycle is already broken by that point.

# Modality is a ``str`` Enum (``class Modality(str, Enum)``), compare to
# the string value directly to avoid importing the enum class itself.
_MODALITY_IMAGE = "image"
_MODALITY_VIDEO = "video"
_MODALITY_AUDIO = "audio"


def _make_sampling_params(
    *,
    num_frames: int = 81,
    init_same_noise: bool = False,
    num_inference_steps: int = 28,
) -> SamplingParams:
    return SamplingParams(
        num_inference_steps=num_inference_steps,
        guidance_scale=5.0,
        height=480,
        width=832,
        num_frames=num_frames,
        seed=42,
        num_samples_per_prompt=1,
        init_same_noise=init_same_noise,
        sde_config=SDEConfig(eta=0.7),
    )


def _make_prompts(n: int) -> Prompts:
    return Prompts.from_unique_prompts(prompts=[f"v{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# _build_diffusion_stage_params — passthrough (Codex finding 3)
#
# Pre-fix this builder emitted 8 hardcoded keys and silently dropped
# ``num_frames`` / ``init_same_noise``: video recipes would get the WAN
# 81-frame default regardless of what the recipe specified. The tests
# below lock the passthrough contract so a future regression is caught
# at unit-test time, not at GPU-smoke time.
# ---------------------------------------------------------------------------


def test_stage_params_emits_num_frames():
    from diffusionrl.rollout.new_pipeline import _build_diffusion_stage_params

    sp = _make_sampling_params(num_frames=33)
    out = _build_diffusion_stage_params(sampling_params=sp, samples_per_prompt=1, sde_indices=None)
    assert out["num_frames"] == 33


def test_stage_params_emits_init_same_noise():
    from diffusionrl.rollout.new_pipeline import _build_diffusion_stage_params

    sp = _make_sampling_params(init_same_noise=True)
    out = _build_diffusion_stage_params(sampling_params=sp, samples_per_prompt=4, sde_indices=[0, 1, 2])
    assert out["init_same_noise"] is True
    # Two-key compat: ``samples_per_prompt`` is what v2 DiffusionParams
    # consume (the previous test locked only the legacy ``num_*`` key
    # and silently let v2 Pipelines drop the value).
    assert out["samples_per_prompt"] == 4
    assert out["num_samples_per_prompt"] == 4
    assert out["sde_indices"] == [0, 1, 2]


def test_stage_params_does_not_emit_noise_group_ids():
    """``noise_group_ids`` is intentionally NOT in stage_params.

    Emitting per-sample group ids via ``stage_params`` (a ``shared_field``
    on ``RolloutReq``) is not slice/chunk-safe: ``RolloutReq.slice`` /
    ``select`` does not slice shared fields, so multi-actor sharding +
    actor-side ``forward_batch_size`` chunking would let every shard /
    chunk see the full-batch ids while its own batch is the chunk
    subset — triggering ``generate_shared_noise``'s
    ``len(noise_group_ids) == batch_size`` hard assert. The clean route
    is ``request_conditions["initial_latents"]`` (CONCAT field,
    auto-sliced); wiring that into v2 Pipelines is a framework
    follow-up. This test pins the contract so a future regression that
    re-adds the broken ``stage_params['noise_group_ids']`` is caught
    immediately.
    """
    from diffusionrl.rollout.new_pipeline import _build_diffusion_stage_params

    sp = _make_sampling_params(init_same_noise=True)
    out = _build_diffusion_stage_params(sampling_params=sp, samples_per_prompt=4, sde_indices=None)
    assert "noise_group_ids" not in out


def test_stage_params_image_path_regression():
    """Image models (no num_frames-driven shape) still get SD3-style fields."""
    from diffusionrl.rollout.new_pipeline import _build_diffusion_stage_params

    sp = _make_sampling_params()
    out = _build_diffusion_stage_params(sampling_params=sp, samples_per_prompt=1, sde_indices=None)
    assert out["height"] == 480
    assert out["width"] == 832
    assert out["num_inference_steps"] == 28
    assert out["guidance_scale"] == pytest.approx(5.0)
    assert out["eta"] == pytest.approx(0.7)
    assert out["seed"] == 42
    assert out["sde_indices"] is None


# ---------------------------------------------------------------------------
# compute_initial_noise_for_request — video latent shape route
#
# Pre-fix the modality switch was ``if modality != "sd35_t2i": return None``,
# so video models lost driver-side noise precompute and inherited the
# engine-side fallback (which loses init_same_noise determinism). After the
# Layer 2 fix, ``wan21_t2v`` / ``wan22_t2v`` return a per-sample noise
# tensor with the right WAN VAE latent shape.
# ---------------------------------------------------------------------------


class _FakeOmegaDict(dict):
    """Minimal DictConfig-like surface — supports ``.get(key, default)``."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeCfg:
    """Minimal cfg shape with ``cfg.rollout.engine.get('modality')``."""

    def __init__(self, modality: Optional[str]) -> None:
        engine = _FakeOmegaDict({"modality": modality})
        self.rollout = type("R", (), {"engine": engine})()


def test_initial_noise_wan21_t2v_emits_video_latent_shape():
    from diffusionrl.rollout.new_pipeline import compute_initial_noise_for_request

    # num_frames=33 → latent_t = (33-1)//4 + 1 = 9
    sp = _make_sampling_params(num_frames=33)
    noise = compute_initial_noise_for_request(
        cfg=_FakeCfg("wan21_t2v"),
        prompts=_make_prompts(2),
        sampling_spec=sp,
        samples_per_prompt=1,
        rollout_id=0,
    )
    assert noise is not None
    # Batch=2 from prompts; channels=16; latent_t=9; H//8=60; W//8=104.
    assert noise.shape == (2, 16, 9, 60, 104), f"got {tuple(noise.shape)}"


def test_initial_noise_wan22_t2v_emits_video_latent_shape():
    from diffusionrl.rollout.new_pipeline import compute_initial_noise_for_request

    # num_frames=5 → latent_t = (5-1)//4 + 1 = 2
    sp = _make_sampling_params(num_frames=5)
    noise = compute_initial_noise_for_request(
        cfg=_FakeCfg("wan22_t2v"),
        prompts=_make_prompts(3),
        sampling_spec=sp,
        samples_per_prompt=1,
        rollout_id=0,
    )
    assert noise is not None
    assert noise.shape == (3, 16, 2, 60, 104)


def test_initial_noise_sd3_path_unchanged():
    """Regression: SD3.5 route still produces (16, H//8, W//8) per sample."""
    from diffusionrl.rollout.new_pipeline import compute_initial_noise_for_request

    sp = _make_sampling_params()
    noise = compute_initial_noise_for_request(
        cfg=_FakeCfg("sd35_t2i"),
        prompts=_make_prompts(2),
        sampling_spec=sp,
        samples_per_prompt=1,
        rollout_id=0,
    )
    assert noise is not None
    assert noise.shape == (2, 16, 60, 104)


def test_initial_noise_unsupported_modality_returns_none():
    """Modalities we haven't wired (HI3 t2i / future) keep the
    rollout-engine-side fallback by returning None."""
    from diffusionrl.rollout.new_pipeline import compute_initial_noise_for_request

    noise = compute_initial_noise_for_request(
        cfg=_FakeCfg("t2i"),
        prompts=_make_prompts(2),
        sampling_spec=_make_sampling_params(),
        samples_per_prompt=1,
        rollout_id=0,
    )
    assert noise is None


# ---------------------------------------------------------------------------
# LatentSegment.modality — survives Batched.select/slice/clone/concat
#
# Pre-fix: ``modality`` was a ``ClassVar[Modality] = Modality.IMAGE`` and
# the ``make_*_segment`` factories stamped it as an instance attribute via
# ``object.__setattr__``. ``Batched.select/slice/clone/concat`` rebuild
# instances with ``type(self)(**kwargs)`` walking declared dataclass
# fields only — the instance-attribute stamp was wiped on every Batched
# op, silently reverting the modality to ``Modality.IMAGE``.
# After the fix, ``modality`` is a real ``shared_field`` dataclass field
# so Batched ops propagate it.
# ---------------------------------------------------------------------------


def _make_image_latent_segment(batch: int = 4, num_steps: int = 4):
    from diffusionrl.types.segments.latent import make_image_segment

    return make_image_segment(
        sample_indices=torch.arange(batch, dtype=torch.long),
        positions=torch.zeros(batch, dtype=torch.long),
        latents=torch.zeros(batch, num_steps + 1, 16, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
    )


def _make_video_latent_segment(batch: int = 4, num_steps: int = 4):
    from diffusionrl.types.segments.latent import make_video_segment

    return make_video_segment(
        sample_indices=torch.arange(batch, dtype=torch.long),
        positions=torch.zeros(batch, dtype=torch.long),
        latents=torch.zeros(batch, num_steps + 1, 16, 3, 8, 8),
        sigmas=torch.linspace(1.0, 0.0, num_steps + 1),
        indices=torch.arange(num_steps + 1, dtype=torch.long),
    )


def test_make_video_segment_stamps_modality():
    seg = _make_video_latent_segment()
    assert seg.modality == _MODALITY_VIDEO


def test_make_image_segment_stamps_modality_default():
    seg = _make_image_latent_segment()
    assert seg.modality == _MODALITY_IMAGE


def test_video_modality_survives_select():
    """``RolloutResp.split()`` calls ``select`` per group — modality must
    NOT revert to IMAGE after that rebuild."""
    seg = _make_video_latent_segment()
    sub = seg.select(torch.tensor([1, 0]))
    assert sub.modality == _MODALITY_VIDEO


def test_video_modality_survives_slice():
    seg = _make_video_latent_segment()
    sub = seg.slice(0, 2)
    assert sub.modality == _MODALITY_VIDEO


def test_video_modality_survives_clone():
    """``_shift_sample_indices`` uses ``clone()`` — modality must propagate."""
    seg = _make_video_latent_segment()
    sub = seg.clone()
    assert sub.modality == _MODALITY_VIDEO


def test_video_modality_survives_concat():
    from diffusionrl.types.segments.latent import LatentSegment

    a = _make_video_latent_segment(batch=2)
    b = _make_video_latent_segment(batch=3)
    merged = LatentSegment.concat([a, b])
    assert merged.modality == _MODALITY_VIDEO


# ---------------------------------------------------------------------------
# _extract_videos_from_output — accept list[Tensor] from the new bridge
#
# Pre-fix: only accepted torch.Tensor; the new
# ``resp_to_samples`` emits ``decoded_videos = list[Tensor]`` (per-sample
# [C, T, H, W]), so the reward pipeline raised "Sampler output did not
# include decoded video media" before advantages/training could run.
# ---------------------------------------------------------------------------


def test_extract_videos_accepts_list_of_4d_tensors():
    """The new bridge emits ``list[Tensor]``; reward pipeline must handle it."""
    from diffusionrl.reward.pipeline import _extract_videos_from_output
    from diffusionrl.types.sample import RolloutSamples

    samples = RolloutSamples.__new__(RolloutSamples)  # bypass __init__ for a stub
    samples.decoded_videos = [
        torch.zeros(3, 5, 8, 8),
        torch.ones(3, 5, 8, 8),
    ]
    out = _extract_videos_from_output(samples)
    assert len(out) == 2
    assert out[0].shape == (3, 5, 8, 8)
    assert torch.equal(out[1], torch.ones(3, 5, 8, 8))


def test_extract_videos_still_accepts_stacked_5d_tensor():
    """Regression: legacy WAN sampler outputs [B, C, T, H, W] stacked tensor."""
    from diffusionrl.reward.pipeline import _extract_videos_from_output
    from diffusionrl.types.sample import RolloutSamples

    samples = RolloutSamples.__new__(RolloutSamples)
    samples.decoded_videos = torch.zeros(4, 3, 5, 8, 8)
    out = _extract_videos_from_output(samples)
    assert len(out) == 4
    assert out[0].shape == (3, 5, 8, 8)


def test_extract_videos_rejects_non_tensor_list_entry():
    """List elements must be 4D tensors; anything else is a contract bug."""
    from diffusionrl.reward.pipeline import _extract_videos_from_output
    from diffusionrl.types.sample import RolloutSamples

    samples = RolloutSamples.__new__(RolloutSamples)
    samples.decoded_videos = [torch.zeros(3, 5, 8, 8), "not-a-tensor"]
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        _extract_videos_from_output(samples)


def test_extract_videos_rejects_wrong_dim_list_entry():
    from diffusionrl.reward.pipeline import _extract_videos_from_output
    from diffusionrl.types.sample import RolloutSamples

    samples = RolloutSamples.__new__(RolloutSamples)
    samples.decoded_videos = [torch.zeros(5, 8, 8)]  # 3D, not 4D
    with pytest.raises(ValueError, match=r"must be 4D"):
        _extract_videos_from_output(samples)
