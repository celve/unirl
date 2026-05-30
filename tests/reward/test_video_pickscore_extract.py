"""Tests for ``VideoPickScoreScorer._extract_first_frame``.

These regression-lock the WAN T2V case where small ``num_frames``
(e.g. 3) produces a per-item video tensor of shape ``(3, 1, H, W)``:
both leading dims are channel-like, so the previous heuristic raised
``ValueError("Ambiguous 4D video tensor shape ...")``, which was then
silently swallowed by ``LocalRewardBackend.compute_rewards`` and
became all-zero rewards.

CPU-only: no CUDA, no HF model download.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from diffusionrl.reward.local.video_pickscore import VideoPickScoreScorer


def test_extract_first_frame_wan_small_t() -> None:
    """WAN T2V with num_frames=3 produces per-item shape (3, 1, H, W)."""
    video = torch.rand(3, 1, 64, 64)
    frame = VideoPickScoreScorer._extract_first_frame(video)
    assert isinstance(frame, Image.Image)
    assert frame.size == (64, 64)


def test_extract_first_frame_rejects_unknown_channel_count() -> None:
    """Unknown channel count must raise loudly, not silently score the wrong axis."""
    video = torch.rand(7, 1, 64, 64)
    with pytest.raises(ValueError, match=r"channel-first \(C, T, H, W\)"):
        VideoPickScoreScorer._extract_first_frame(video)
