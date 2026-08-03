"""MiniMax-H3 -- 33B omni-modal joint video + stereo audio diffusion.

Covers ``t2va`` (text -> video+audio) and ``fl2va`` (keyframe conditioning) on
the FL2VA checkpoint, plus the reference encoding and packing layer of
``ref2va`` on the Ref2VA checkpoint (``transformer_subfolder`` selects the
partition). ref2va's pipeline assembly -- turning UniRL primitives into
prepared references and building the labelled prompt presentation -- is not
wired yet; see the branch notes.
"""

from .bundle import MiniMaxH3Bundle
from .conditions import MiniMaxH3Conditions
from .config import (
    MINIMAX_H3_LATENT_CHANNELS,
    MINIMAX_H3_PATCH_SIZE,
    MINIMAX_H3_SPATIAL_COMPRESSION,
    MINIMAX_H3_TEMPORAL_COMPRESSION,
    MiniMaxH3PipelineConfig,
)
from .diffusion import MiniMaxH3DiffusionStage, MiniMaxH3DiffusionStep
from .keyframe import (
    MiniMaxH3KeyframeEncodeStage,
    prepare_keyframes,
    resolve_keyframe_anchors,
)
from .packing import (
    MiniMaxH3Geometry,
    build_layout,
    build_ref2va_layout,
    build_t2va_layout,
    row_timestep_plan,
)
from .pipeline import MiniMaxH3Pipeline
from .reference import (
    MiniMaxH3ReferenceEncodeStage,
    decode_reference_geometry,
    encode_reference_geometry,
    validate_references,
)
from .text_embed import MiniMaxH3TextEmbedStage
from .vae import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MiniMaxH3AudioDecodeStage,
    MiniMaxH3VideoDecodeStage,
)

__all__ = [
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MINIMAX_H3_LATENT_CHANNELS",
    "MINIMAX_H3_PATCH_SIZE",
    "MINIMAX_H3_SPATIAL_COMPRESSION",
    "MINIMAX_H3_TEMPORAL_COMPRESSION",
    "MiniMaxH3AudioDecodeStage",
    "MiniMaxH3Bundle",
    "MiniMaxH3Conditions",
    "MiniMaxH3DiffusionStage",
    "MiniMaxH3DiffusionStep",
    "MiniMaxH3Geometry",
    "MiniMaxH3KeyframeEncodeStage",
    "MiniMaxH3Pipeline",
    "MiniMaxH3PipelineConfig",
    "MiniMaxH3TextEmbedStage",
    "MiniMaxH3VideoDecodeStage",
    "MiniMaxH3ReferenceEncodeStage",
    "build_layout",
    "build_ref2va_layout",
    "build_t2va_layout",
    "decode_reference_geometry",
    "encode_reference_geometry",
    "prepare_keyframes",
    "resolve_keyframe_anchors",
    "row_timestep_plan",
    "validate_references",
]
