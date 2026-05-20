"""Fail-fast on WAN VAE frame-count invariant ``(num_frames - 1) % 4 == 0``.

The WAN VAE (``AutoencoderKLWan``) temporally downsamples by 4 with the
``+1`` reference-frame offset, so any pixel ``num_frames`` that doesn't
satisfy ``(num_frames - 1) % 4 == 0`` rounds down via integer division
and silently produces a mismatched latent shape. Each WAN21/22 stage
and pipeline ``latent_shape`` entrypoint must raise ``ValueError``
*before* the floor-divide.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from diffusionrl.models_new.wan21.diffusion import WAN21DiffusionStage
from diffusionrl.models_new.wan21.pipeline import WAN21Pipeline
from diffusionrl.models_new.wan22.diffusion import WAN22DiffusionStage
from diffusionrl.models_new.wan22.pipeline import WAN22Pipeline

# (num_frames, expected_latent_t) — all satisfy (n-1) % 4 == 0.
VALID_CASES = [(1, 1), (5, 2), (9, 3), (17, 5), (81, 21)]
INVALID_FRAMES = [2, 3, 4, 6, 7, 8, 10, 80]


def _stage(cls):
    """Build a stage without running ``__init__`` (which needs a real bundle)."""
    obj = cls.__new__(cls)
    obj.vae_scale_factor = cls._SPATIAL_DOWNSAMPLE
    obj.temporal_scale_factor = cls._TEMPORAL_DOWNSAMPLE
    obj.latent_channels = cls._DEFAULT_LATENT_CHANNELS
    return obj


@pytest.mark.parametrize("cls", [WAN21DiffusionStage, WAN22DiffusionStage])
@pytest.mark.parametrize("bad_frames", INVALID_FRAMES)
def test_stage_latent_shape_rejects_bad_frame_count(cls, bad_frames):
    stage = _stage(cls)
    with pytest.raises(ValueError, match=r"num_frames - 1\) % 4 == 0"):
        stage._latent_shape(num_frames=bad_frames, height=480, width=832)


@pytest.mark.parametrize("cls", [WAN21DiffusionStage, WAN22DiffusionStage])
@pytest.mark.parametrize("num_frames,latent_t", VALID_CASES)
def test_stage_latent_shape_accepts_valid_frame_count(cls, num_frames, latent_t):
    stage = _stage(cls)
    shape = stage._latent_shape(num_frames=num_frames, height=480, width=832)
    assert shape == (16, latent_t, 60, 104)


@pytest.mark.parametrize("cls", [WAN21Pipeline, WAN22Pipeline])
@pytest.mark.parametrize("bad_frames", INVALID_FRAMES)
def test_pipeline_latent_shape_rejects_bad_frame_count(cls, bad_frames):
    spec = SimpleNamespace(height=480, width=832, num_frames=bad_frames)
    with pytest.raises(ValueError, match=r"num_frames - 1\) % 4 == 0"):
        cls.latent_shape(model_config=None, sampling_spec=spec)


@pytest.mark.parametrize("cls", [WAN21Pipeline, WAN22Pipeline])
@pytest.mark.parametrize("num_frames,latent_t", VALID_CASES)
def test_pipeline_latent_shape_accepts_valid_frame_count(cls, num_frames, latent_t):
    spec = SimpleNamespace(height=480, width=832, num_frames=num_frames)
    shape = cls.latent_shape(model_config=None, sampling_spec=spec)
    assert shape == (16, latent_t, 60, 104)
