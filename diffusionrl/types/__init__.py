"""
Cross-module data types for diffusionrl.

This package provides shared dataclasses and validation helpers used by:
- rollout control-plane
- ray actors
- samplers and losses
"""

from .reward import RewardRequest, RewardResponse, RewardType
from .sampling import (
    InferenceRequest,
    LogProbData,
    PromptEmbeddings,
    SamplerOutput,
    SampleStatus,
)
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
    "InferenceRequest",
    "LogProbData",
    "ForwardTrainingBatch",
    "PromptEmbeddings",
    "RewardRequest",
    "RewardResponse",
    "RewardType",
    "SampleStatus",
    "SamplerOutput",
    "TimestepData",
    "TrainingBatch",
    "is_backward_batch",
    "is_forward_batch",
]
