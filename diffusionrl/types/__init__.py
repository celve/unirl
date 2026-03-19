"""
Cross-module data types for diffusionrl.

This package provides shared dataclasses and validation helpers used by:
- rollout control-plane
- ray actors
- samplers and losses
"""

from .reward import RewardRequest, RewardResponse, RewardType
from .engine import EngineCapabilities, EngineConfig
from .sampling import (
    RolloutOutput,
    RolloutRequest,
    LogProbData,
    PromptEmbeddings,
    SamplingRequirements,
)
from .sde import SDEConfig, SDEScheduleConfig
from .training_batch import (
    BackwardTrainingBatch,
    ForwardTrainingBatch,
    TimestepData,
    TrainingBatch,
    is_backward_batch,
    is_forward_batch,
)

__all__ = [
    "BackwardTrainingBatch",
    "EngineCapabilities",
    "EngineConfig",
    "RolloutOutput",
    "RolloutRequest",
    "LogProbData",
    "ForwardTrainingBatch",
    "PromptEmbeddings",
    "RewardRequest",
    "RewardResponse",
    "RewardType",
    "SamplingRequirements",
    "SDEConfig",
    "SDEScheduleConfig",
    "TimestepData",
    "TrainingBatch",
    "is_backward_batch",
    "is_forward_batch",
]
