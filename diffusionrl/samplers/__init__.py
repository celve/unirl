"""
Samplers for GRPO training.

These samplers generate trajectories and log probabilities for policy gradient training.

Available engines:
- Direct sampling engines: native PyTorch sampler paths that run on TrainingActor
- SGLang: dedicated rollout-side engine for separated training/inference

Engine selection:
- Use sampler/model defaults for direct sampling on training actors
- Use SGLang when rollout is intentionally split into a dedicated service

Engine Interface:
Dedicated rollout engines implement BaseRolloutEngine for unified Ray actor integration:
- initialize(): Load models and setup
- generate(): Generate samples with log_probs
- encode_prompt(): Text encoding
- update_weights(): Sync weights from training
- sleep()/wake_up(): Memory/runtime lifecycle
"""

# Base classes and utilities
from diffusionrl.sde.runtime import (
    compute_sde_log_prob,
    get_sigma_schedule,
    sde_step_with_log_prob,
)
from .base import BaseSampler, RolloutOutput
from .schedulers import (
    TimestepScheduler,
    AllSDEScheduler,
    WindowScheduler,
    WindowConfig,
    get_scheduler,
)

# Engine interface
from .engine import (
    BaseRolloutEngine,
    ENGINE_REGISTRY,
    register_engine,
    get_engine,
    create_engine,
)

# Native PyTorch samplers used by training-actor direct sampling
from .fsdp import FluxSampler, SD3Sampler, FSDPHunyuanSampler

# SGLang dedicated rollout engine
from .sglang import SGLangRolloutEngine

__all__ = [
    # Log probability computation
    "compute_sde_log_prob",
    "get_sigma_schedule",
    "sde_step_with_log_prob",
    # Base classes
    "BaseSampler",
    "RolloutOutput",
    # Engine interface
    "BaseRolloutEngine",
    "ENGINE_REGISTRY",
    "register_engine",
    "get_engine",
    "create_engine",
    # Native PyTorch samplers
    "FluxSampler",
    "SD3Sampler",
    "FSDPHunyuanSampler",
    # SGLang dedicated rollout engine
    "SGLangRolloutEngine",
    # Timestep schedulers (MixGRPO)
    "TimestepScheduler",
    "AllSDEScheduler",
    "WindowScheduler",
    "WindowConfig",
    "get_scheduler",
]
