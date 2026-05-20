"""diffusionrl stage-driven algorithms.

Public surface for the ``models_new`` training contract.
"""

from __future__ import annotations

from .base import AlgorithmStepResult, StageAlgorithm
from .grpo import ARGRPO, DiffusionGRPO
from .nft import DiffusionNFT, DiffusionNFTConfig
from .rollout_control import (
    GRPORolloutControl,
    GRPORolloutControlConfig,
    NFTRolloutControl,
    NFTRolloutControlConfig,
)

__all__ = [
    "ARGRPO",
    "AlgorithmStepResult",
    "DiffusionGRPO",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "GRPORolloutControl",
    "GRPORolloutControlConfig",
    "NFTRolloutControl",
    "NFTRolloutControlConfig",
    "StageAlgorithm",
]
