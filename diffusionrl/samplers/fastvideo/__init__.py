"""
FastVideo Engine Samplers.

FastVideo-based samplers for efficient video generation in GRPO training.
These samplers use FastVideo for parallel video generation and compute
log probabilities via trajectory replay.

Available samplers:
- FastVideoSampler: FastVideo sampler with trajectory replay for log_prob
- FastVideoSamplerV2: (Future) Native FastVideo log_prob support

Engine:
- FastVideoInferenceEngine: Unified engine interface for Ray actors
"""

from .fastvideo_sampler import FastVideoSampler, FastVideoSamplerV2
from .engine import FastVideoInferenceEngine

__all__ = [
    # Samplers
    "FastVideoSampler",
    "FastVideoSamplerV2",
    # Engine
    "FastVideoInferenceEngine",
]
