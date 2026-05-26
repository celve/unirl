from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from diffusionrl.algorithms import StageAlgorithm
from diffusionrl.models.types.ar import ARStage
from diffusionrl.models.types.diffusion import DiffusionStage
from diffusionrl.training.ema import EMA

Stage = Union[DiffusionStage, ARStage]


@dataclass
class TrainTrack:
    """Per-track grouping of all training objects.

    Plain data — no delegation methods.  The training loop accesses
    ``track.stage``, ``track.ema``, ``track.optimizer`` directly.
    """

    stage: Stage
    ema: Optional[EMA]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    algorithm: StageAlgorithm
    micro_batch_size: int


__all__ = ["TrainTrack", "Stage"]
