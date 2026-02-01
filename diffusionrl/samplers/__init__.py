"""
Samplers for GRPO training.

These samplers generate trajectories and log probabilities for policy gradient training.

Available engines:
- FSDP: Native PyTorch samplers compatible with FSDP (Fully Sharded Data Parallel)
- FastVideo: FastVideo-based samplers for efficient video generation
- SGLang: Reserved for future SGLang-based samplers

Engine selection:
- Use FSDP engine for image models (FLUX, SD3) and when DanceGRPO alignment is needed
- Use FastVideo engine for high-throughput video generation
- Use SGLang engine (future) for distributed inference

Engine Interface:
All engines implement BaseInferenceEngine for unified Ray actor integration:
- initialize(): Load models and setup
- generate(): Generate samples with log_probs
- encode_prompt(): Text encoding
- update_weights(): Sync weights from training
- offload()/onload(): Memory management
"""

# Base classes and utilities
from .log_prob import compute_sde_log_prob, get_sigma_schedule, sde_step_with_log_prob
from .base import BaseSampler, SamplerOutput, TrajectoryReplaySampler
from .schedulers import (
    TimestepScheduler,
    AllSDEScheduler,
    WindowScheduler,
    WindowConfig,
    get_scheduler,
)

# Engine interface
from .engine import (
    BaseInferenceEngine,
    EngineConfig,
    ENGINE_REGISTRY,
    register_engine,
    get_engine,
    create_engine,
)

# FSDP Engine (native PyTorch, DanceGRPO-aligned)
from .fsdp import FluxSampler, SD3Sampler, FSDPHunyuanSampler, FSDPInferenceEngine

# FastVideo Engine
from .fastvideo import FastVideoSampler, FastVideoSamplerV2, FastVideoInferenceEngine

# SGLang Engine (placeholder)
from .sglang import SGLangInferenceEngine

__all__ = [
    # Log probability computation
    "compute_sde_log_prob",
    "get_sigma_schedule",
    "sde_step_with_log_prob",
    # Base classes
    "BaseSampler",
    "SamplerOutput",
    "TrajectoryReplaySampler",
    # Engine interface
    "BaseInferenceEngine",
    "EngineConfig",
    "ENGINE_REGISTRY",
    "register_engine",
    "get_engine",
    "create_engine",
    # FSDP Engine
    "FluxSampler",
    "SD3Sampler",
    "FSDPHunyuanSampler",
    "FSDPInferenceEngine",
    # FastVideo Engine
    "FastVideoSampler",
    "FastVideoSamplerV2",
    "FastVideoInferenceEngine",
    # SGLang Engine
    "SGLangInferenceEngine",
    # Timestep schedulers (MixGRPO)
    "TimestepScheduler",
    "AllSDEScheduler",
    "WindowScheduler",
    "WindowConfig",
    "get_scheduler",
]
