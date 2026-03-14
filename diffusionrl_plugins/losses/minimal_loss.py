"""Minimal custom loss plugin templates.

Copy one of the classes below and replace the dummy objective.

.. note::
    Loss computation is now unified into the algorithm classes
    (``diffusionrl.algorithms``).  These templates still work as standalone
    loss plugins loaded via ``--algorithm.loss-path``, but for new code
    consider extending ``BaseAlgorithm`` directly (see
    ``diffusionrl_plugins/algorithms/minimal_algorithm.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.types import ForwardTrainingBatch, PromptEmbeddings, TimestepData


class MinimalBackwardLoss:
    """Template for trajectory/log-prob based losses (GRPO-like)."""

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    def __init__(self, scale: float = 1.0, **kwargs: Any) -> None:
        self.scale = float(scale)
        self.extra_kwargs = dict(kwargs)

    def compute_timestep(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Replace with your real objective.
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"loss_scale": self.scale}


class MinimalForwardLoss:
    """Template for clean-latent forward losses (NFT-like)."""

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        return {
            "requires_trajectory": False,
            "requires_log_prob": False,
            "requires_embeddings": True,
        }

    def __init__(self, scale: float = 1.0, **kwargs: Any) -> None:
        self.scale = float(scale)
        self.extra_kwargs = dict(kwargs)

    def compute_batch(
        self,
        model: nn.Module,
        batch: ForwardTrainingBatch,
        timestep_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Replace with your real objective.
        loss = batch.clean_latents.float().sum() * 0.0
        return loss, {"loss_scale": self.scale}
