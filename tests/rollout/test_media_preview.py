"""Tests for ``build_media_preview_for_track`` — track-aware preview builder.

Replaces the legacy ``RolloutResponse.attach_media_preview`` tests that
read decoded media off ``samples.decoded_images`` / ``decoded_videos``.
The new helper consumes ``track.decoded`` (``Images`` or ``Videos``)
plus ``req.primitives['text']`` directly.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from diffusionrl.types.media_preview import (
    MediaPreview,
    build_media_preview_for_track,
)
from diffusionrl.types.primitives import (
    Image,
    Images,
    Texts,
    Video,
    Videos,
)
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutTrack


def _make_track(*, decoded, rewards: List[float], n: int) -> RolloutTrack:
    return RolloutTrack(
        sample_ids=[f"s{i}" for i in range(n)],
        parent_ids=["g"] * n,
        decoded=decoded,
        rewards=torch.tensor(rewards, dtype=torch.float32) if rewards else None,
    )


def _make_req(texts: List[str]) -> RolloutReq:
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(texts))],
        group_ids=["g"] * len(texts),
        primitives={"text": Texts(texts=list(texts))},
    )


def test_image_path_emits_pil_per_sample():
    images = Images.from_list([Image(pixels=torch.rand(3, 4, 4)) for _ in range(3)])
    track = _make_track(decoded=images, rewards=[0.1, 0.2, 0.3], n=3)
    req = _make_req(["p0", "p1", "p2"])

    preview = build_media_preview_for_track(req=req, track=track, max_items=8)
    assert isinstance(preview, MediaPreview)
    assert len(preview) == 3
    assert preview.prompts == ["p0", "p1", "p2"]
    assert preview.rewards == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert not preview.videos


def test_image_path_caps_at_max_items():
    images = Images.from_list([Image(pixels=torch.rand(3, 4, 4)) for _ in range(5)])
    track = _make_track(decoded=images, rewards=[0.0, 1.0, 2.0, 3.0, 4.0], n=5)
    req = _make_req(["p0", "p1", "p2", "p3", "p4"])

    preview = build_media_preview_for_track(req=req, track=track, max_items=2)
    assert len(preview) == 2
    assert preview.prompts == ["p0", "p1"]
    assert preview.rewards == pytest.approx([0.0, 1.0], abs=1e-6)


def test_image_path_strips_alpha_to_three_channels():
    """Four-channel input is sliced to RGB before PIL conversion."""
    images = Images.from_list([Image(pixels=torch.rand(4, 4, 4)) for _ in range(1)])
    track = _make_track(decoded=images, rewards=[0.5], n=1)
    req = _make_req(["only"])

    preview = build_media_preview_for_track(req=req, track=track, max_items=8)
    assert preview is not None
    # tensor_frame_to_pil returns a PIL.Image; verify by attribute presence.
    img = preview.images[0]
    assert hasattr(img, "mode")
    assert img.mode == "RGB"


def test_video_path_emits_4d_cpu_float32_per_sample():
    """Per-sample [T, C, H, W] frames → preview keeps [C, T, H, W] CPU float32."""
    n = 2
    videos = Videos.from_list([Video(frames=torch.rand(3, 3, 4, 4)) for _ in range(n)])
    track = _make_track(decoded=videos, rewards=[0.1, 0.2], n=n)
    req = _make_req(["v0", "v1"])

    preview = build_media_preview_for_track(req=req, track=track, max_items=8)
    assert isinstance(preview, MediaPreview)
    assert len(preview) == n
    assert not preview.images
    for vid in preview.videos:
        assert torch.is_tensor(vid)
        assert vid.dim() == 4
        assert vid.dtype == torch.float32
        assert vid.device.type == "cpu"


def test_returns_none_when_decoded_is_none():
    track = _make_track(decoded=None, rewards=[], n=0)
    req = _make_req([])
    assert build_media_preview_for_track(req=req, track=track, max_items=4) is None


def test_returns_none_when_decoded_is_text():
    """Text tracks aren't image/video — preview is N/A."""
    texts = Texts(texts=["x", "y"])
    track = _make_track(decoded=texts, rewards=[0.1, 0.2], n=2)
    req = _make_req(["p0", "p1"])
    assert build_media_preview_for_track(req=req, track=track, max_items=4) is None


def test_zero_rewards_use_default_zero_floats():
    """Missing/empty rewards: emitted rewards default to 0.0 per sample."""
    images = Images.from_list([Image(pixels=torch.rand(3, 4, 4)) for _ in range(2)])
    track = _make_track(decoded=images, rewards=[], n=2)  # rewards stays None
    req = _make_req(["p0", "p1"])

    preview = build_media_preview_for_track(req=req, track=track, max_items=4)
    assert preview is not None
    assert preview.rewards == [0.0, 0.0]
