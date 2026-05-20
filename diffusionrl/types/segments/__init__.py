"""Segment types — SoA batched containers for generation outputs.

A ``Segment`` is always batched (SoA); per-sample access is via raw
indexing on ``sample_indices``. Each modality has its own subclass:

- ``LatentSegment`` covers image / video / audio diffusion rollouts.
- ``TextSegment`` covers AR token rollouts (varlen-packed).
"""

from __future__ import annotations

from diffusionrl.types.segments.base import Segment, SegmentStatus
from diffusionrl.types.segments.latent import (
    LatentSegment,
    make_audio_segment,
    make_image_segment,
    make_video_segment,
)
from diffusionrl.types.segments.text import TextSegment

__all__ = [
    "LatentSegment",
    "Segment",
    "SegmentStatus",
    "TextSegment",
    "make_audio_segment",
    "make_image_segment",
    "make_video_segment",
]
