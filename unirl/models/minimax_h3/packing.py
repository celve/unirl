"""Geometry resolution and packed-row layout for MiniMax-H3.

MiniMax-H3 runs its transformer over ONE packed 1-D sequence holding every
modality at once. The row order depends on the task::

    t2va   [ text | target audio | target video ]
    fl2va  [ text | keyframe conditions | target audio | target video ]
    ref2va [ text | reference blocks    | target audio | target video ]

In every case the conditioning rows are a PREFIX of their modality's rows and
are never stepped, so the generated-row counts are identical across all three
tasks -- which is what keeps ``LatentSegment`` task-agnostic.

This module is a thin resolver over the vendored builders -- the row geometry,
the float64 rotary clock and the tag values are checkpoint contracts, so they
are USED from ``vendor.packing`` rather than reimplemented. What lives here is
only the mapping from UniRL's ``DiffusionSamplingParams`` to the arguments those
builders want, plus the sigma -> timestep conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .config import (
    MINIMAX_H3_LATENT_CHANNELS,
    MINIMAX_H3_PATCH_SIZE,
    MINIMAX_H3_SPATIAL_COMPRESSION,
)
from .vendor import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    resolve_canvas_size,
    video_latent_num_frames,
)
from .vendor.packing_ref2va import build_ref2va_packed_sequence

# MiniMax-H3's "clean" end of the timestep axis. Its convention is t = 1 - sigma
# with t = 1 meaning fully denoised, which is the opposite direction to every
# other model in this repo -- hence a name rather than a bare 1.0 at the call site.
MINIMAX_H3_CLEAN_TIMESTEP = 1.0


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """The resolved shape of one request. Depends only on SHARED sampling params.

    Every field here is a pure function of ``(aspect, duration)``, never of
    per-sample data. That is load-bearing: ``LatentSegment`` stores latents in a
    ``FieldKind.CONCAT`` tensor, so ``torch.cat`` across the batch requires
    identical non-batch dims. Per-sample geometry would need a varlen segment
    UniRL does not have.
    """

    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int

    @property
    def rows_per_frame(self) -> int:
        _, patch_h, patch_w = MINIMAX_H3_PATCH_SIZE
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * self.rows_per_frame

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    @property
    def video_token_dim(self) -> int:
        patch_t, patch_h, patch_w = MINIMAX_H3_PATCH_SIZE
        return MINIMAX_H3_LATENT_CHANNELS * patch_t * patch_h * patch_w

    @property
    def latent_shape(self) -> Tuple[int, int, int, int]:
        """Per-sample UNPACKED video latent shape ``(C, T_lat, H_lat, W_lat)``."""
        return (
            MINIMAX_H3_LATENT_CHANNELS,
            self.num_latent_frames,
            self.latent_height,
            self.latent_width,
        )

    @classmethod
    def resolve(cls, *, height: int, width: int, num_frames: int) -> "MiniMaxH3Geometry":
        """Validate a requested ``(height, width, num_frames)`` against H3.

        MiniMax-H3 does NOT accept an arbitrary canvas. It was released for a
        768 pixel SHORT EDGE only, both axes a multiple of 32, area soft-capped
        at 768x1344 -- so the canvas is a function of the aspect ratio alone.
        Frame count is likewise snapped: the video VAE encodes 17 pixel frames
        per chunk and drops 3 trailing latent frames, so only ``17n + 5`` counts
        round-trip, and duration is clamped to [5, 15]s at a fixed 24 fps.

        This raises rather than silently re-resolving. A recipe asking for
        512x768 is not a slightly-off request that should be rounded -- it is
        outside what the checkpoint can do, and quietly training at a different
        resolution than the recipe states is exactly the kind of drift that is
        impossible to spot later. The error carries the nearest legal values.
        """
        canvas_height, canvas_width = resolve_canvas_size(float(width), float(height))
        if (int(height), int(width)) != (canvas_height, canvas_width):
            raise ValueError(
                f"MiniMaxH3Geometry: height={height} width={width} is not a MiniMax-H3 canvas. The model was "
                f"released for a 768 pixel short edge with both axes a multiple of 32; for this aspect ratio the "
                f"only legal canvas is height={canvas_height} width={canvas_width}. There is no smaller setting "
                f"for a cheaper run -- 768x768 for 5s (~22k packed rows) is the floor."
            )
        aligned = align_num_frames(int(num_frames))
        if aligned != int(num_frames):
            raise ValueError(
                f"MiniMaxH3Geometry: num_frames={num_frames} does not round-trip through the video VAE, which maps "
                f"`17n + 5` pixel frames to `5n + 2` latent frames. Nearest legal value: {aligned} "
                f"({aligned / MINIMAX_H3_FPS:.2f}s at {MINIMAX_H3_FPS} fps; H3 supports 5-15s)."
            )
        return cls(
            height=canvas_height,
            width=canvas_width,
            num_frames=aligned,
            num_latent_frames=video_latent_num_frames(aligned),
            latent_height=canvas_height // MINIMAX_H3_SPATIAL_COMPRESSION,
            latent_width=canvas_width // MINIMAX_H3_SPATIAL_COMPRESSION,
            num_audio_latents=audio_latent_num_frames(aligned),
        )

    @classmethod
    def from_params(cls, params) -> "MiniMaxH3Geometry":
        """Resolve from a ``DiffusionSamplingParams``-shaped object."""
        return cls.resolve(height=int(params.height), width=int(params.width), num_frames=int(params.num_frames))


def build_layout(
    geometry: MiniMaxH3Geometry,
    *,
    text_token_tags: torch.Tensor,
    keyframe_anchors: Tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build the packed layout for a t2va or fl2va request.

    ``keyframe_anchors`` holds one entry per keyframe conditioning block, in
    packed order: ``"first"`` anchors it at the first latent frame, ``"last"``
    at the last. Empty (the t2va case) yields ``[text | audio | video]``.

    ``text_token_tags`` is per-row, not a constant: MiniMax-H3 tags the rows of
    a keyframe's *vision block* -- which lives inside the text stream, since the
    keyframe also goes through the Qwen3-VL conditioner -- as VIDEO rather than
    TEXT. t2va has no vision block and so passes a uniform vector.
    """
    return build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=geometry.num_latent_frames,
        latent_height=geometry.latent_height,
        latent_width=geometry.latent_width,
        num_audio_latents=geometry.num_audio_latents,
        patch_size=MINIMAX_H3_PATCH_SIZE,
        keyframe_anchors=tuple(keyframe_anchors),
    )


def build_ref2va_layout(
    geometry: MiniMaxH3Geometry,
    *,
    text_token_tags: torch.Tensor,
    references,
) -> MiniMaxH3PackedSequence:
    """``[text | reference blocks | target audio | target video]`` for ref2va.

    ``references`` are ``MiniMaxH3PreparedReference`` rows in packed order; only
    their latent geometry is read, which is what lets ``replay`` rebuild the
    layout from the serialized geometry table without touching a VAE.
    """
    return build_ref2va_packed_sequence(
        text_token_tags=text_token_tags,
        references=list(references),
        num_latent_frames=geometry.num_latent_frames,
        latent_height=geometry.latent_height,
        latent_width=geometry.latent_width,
        num_audio_latents=geometry.num_audio_latents,
        patch_size=MINIMAX_H3_PATCH_SIZE,
    )


def build_t2va_layout(geometry: MiniMaxH3Geometry, num_text_tokens: int) -> MiniMaxH3PackedSequence:
    """``[text | audio | video]`` with a uniform text tag -- the t2va case."""
    return build_layout(
        geometry,
        text_token_tags=torch.full((int(num_text_tokens),), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
        keyframe_anchors=(),
    )


def row_timestep_plan(
    layout: MiniMaxH3PackedSequence,
    *,
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
    condition_video_timestep: float = MINIMAX_H3_KEYFRAME_NOISE_AUG,
    condition_audio_timestep: float = MINIMAX_H3_CLEAN_TIMESTEP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(unique_timesteps, timestep_indices)`` for one denoising step.

    MiniMax-H3 conditions its AdaLN on ``t = 1 - sigma`` in ``[0, 1]``, UNSCALED
    (no x1000) and with ``t = 1`` meaning clean -- the opposite direction to
    every other model in this repo. Text rows never reach an output head and
    inherit the video timestep.

    Conditioning rows do NOT step -- they are pinned for every denoising step,
    which is why these are constants rather than schedule entries. The two
    modalities are pinned DIFFERENTLY, and the asymmetry is the model's:

    * **Visual** conditioning (fl2va keyframes, ref2va image/video references)
      is noise-augmented to ``t = 0.999`` -- 99.9% signal, not fully clean.
    * **Audio** conditioning (ref2va reference soundtracks) rides **clean** at
      ``t = 1.0``: it is never noised, and its VAE posterior is taken as the
      mean rather than sampled.

    With no conditioning rows both arguments are inert -- ``torch.unique`` only
    sees the timesteps rows actually carry.
    """
    return build_row_timesteps(
        layout,
        video_timestep=float(1.0 - float(video_sigma)),
        audio_timestep=float(1.0 - float(audio_sigma)),
        condition_video_timestep=float(condition_video_timestep),
        condition_audio_timestep=float(condition_audio_timestep),
    )


__all__ = [
    "MiniMaxH3Geometry",
    "build_layout",
    "build_ref2va_layout",
    "build_t2va_layout",
    "row_timestep_plan",
]
