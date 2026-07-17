"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .base import AlgorithmStepResult, StageAlgorithm
from .cppo import CPPO, CPPOConfig
from .diffusionnft import DiffusionNFT, DiffusionNFTConfig
from .dppo import DPPO, DPPOConfig
from .drpo import DRPO, DRPOConfig
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .flowgrpo import FlowGRPO, FlowGRPOConfig
from .gae import (
    ActionTokenGAE,
    action_token_gae,
    adaptive_policy_lambda,
    compute_action_token_gae,
    compute_skip_observation_gae,
    concatenate_action_tokens,
    split_action_tokens,
)
from .grpo import GRPO, GRPOConfig
from .sao import SAO, SAOConfig, SAOLossOutput, sao_policy_loss
from .value import (
    TokenValueAlgorithm,
    TokenValueConfig,
    ValueAlgorithm,
    ValueConfig,
    ValueLossOutput,
    masked_value_loss,
)

__all__ = [
    "GRPO",
    "GRPOConfig",
    "CPPO",
    "CPPOConfig",
    "DPPO",
    "DPPOConfig",
    "DRPO",
    "DRPOConfig",
    "AlgorithmStepResult",
    "FlowGRPO",
    "FlowGRPOConfig",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
    "SAO",
    "SAOConfig",
    "SAOLossOutput",
    "sao_policy_loss",
    "TokenValueAlgorithm",
    "TokenValueConfig",
    "ValueAlgorithm",
    "ValueConfig",
    "ValueLossOutput",
    "masked_value_loss",
    "ActionTokenGAE",
    "action_token_gae",
    "adaptive_policy_lambda",
    "compute_action_token_gae",
    "compute_skip_observation_gae",
    "concatenate_action_tokens",
    "split_action_tokens",
]
