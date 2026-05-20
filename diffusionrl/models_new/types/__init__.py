"""Model type protocols and interfaces.

Pipeline stages (each ``X → Y`` between tiers):

- ``EncodeStage[P, C]`` — Primitive → Condition (e.g. VAE encode).
- ``EmbedStage[P, C]`` — Primitive → Condition (e.g. text encoder).
- ``DiffusionStage`` — Conditions → LatentSegment.
- ``ARStage`` — Conditions → TextSegment.
- ``DecodeStage[S, P]`` — Segment → Primitive (e.g. VAE decode).

Step kernels (per-step math, tensor I/O):

- ``DiffusionStep`` — single denoising transition.
- ``ARStep`` — single token sample.

All schedule / sampling parameters are passed at call time on the rollout-
level stages (``diffuse(...)`` / ``autoregress(...)``) — Stages are stateless
modulo the model they wrap.
"""

from __future__ import annotations

from diffusionrl.models_new.types.ar import ARSamplingParams, ARStage, ARStep
from diffusionrl.models_new.types.bundle import Bundle
from diffusionrl.models_new.types.codec import DecodeStage, EncodeStage
from diffusionrl.models_new.types.diffusion import DiffusionStage, DiffusionStep
from diffusionrl.models_new.types.embedding import EmbedStage
from diffusionrl.models_new.types.pipeline import Pipeline
from diffusionrl.models_new.types.replay_result import ReplayResult

__all__ = [
    "ARSamplingParams",
    "ARStage",
    "ARStep",
    "Bundle",
    "DecodeStage",
    "DiffusionStage",
    "DiffusionStep",
    "EmbedStage",
    "EncodeStage",
    "Pipeline",
    "ReplayResult",
]
