from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

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

    ``algorithms`` is a slot-keyed dict so one track can host multiple
    StageAlgorithm instances sharing the same optimizer / LoRA wrap.
    Typical PE / SD3 case: one entry whose key equals the track name.
    HI3 shared-backbone case: two entries (e.g. ``{"image":
    DiffusionGRPO, "ar": ARGRPO}``) — both accumulate gradients on the
    same backbone LoRA and a single ``optimizer.step()`` updates them
    jointly (pre-#136 ``StageTrainStack`` semantic).
    """

    stage: Stage
    ema: Optional[EMA]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    algorithms: Dict[str, StageAlgorithm]
    micro_batch_size: int


__all__ = ["TrainTrack", "Stage"]
