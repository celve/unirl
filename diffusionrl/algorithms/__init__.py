"""diffusionrl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .base import AlgorithmStepResult, StageAlgorithm
from .dppo import DiffusionDPPO
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
    "DiffusionDPPO",
    "DiffusionGRPO",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "GRPORolloutControl",
    "GRPORolloutControlConfig",
    "NFTRolloutControl",
    "NFTRolloutControlConfig",
    "StageAlgorithm",
]
