"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .ar_grpo import ARGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .diffusion_grpo import DiffusionGRPO
from .dppo import DiffusionDPPO
from .nft import DiffusionNFT, DiffusionNFTConfig
from .spo_dppo import ARSPODPPO

__all__ = [
    "ARGRPO",
    "ARSPODPPO",
    "AlgorithmStepResult",
    "DiffusionDPPO",
    "DiffusionGRPO",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "StageAlgorithm",
]
