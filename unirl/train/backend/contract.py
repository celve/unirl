"""Structural contract for a training backend.

:class:`TrainBackend` documents the method/attribute surface the training
stacks (and the trainer / weight-sync / algorithm siblings) rely on, so a
second backend can be added beside
:class:`~unirl.train.backend.fsdp.FSDPBackend` without threading a concrete
type through the stack.

It is a plain ``Protocol`` — structural, never ``isinstance``-checked at runtime
(callers hold ``Remote`` ``Handle`` proxies, not the concrete object), so it is
static documentation only and is not inherited by the concrete backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from unirl.train.ema import EMA


class TrainBackend(Protocol):
    """Method/attribute surface a training backend exposes to the stacks."""

    # Training state owned by the backend.
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    ema: Optional["EMA"]
    # Compute device of the trainable module. Private name preserved because the
    # unified-model stack reaches it directly; revisit when a second backend lands.
    _device: torch.device

    @property
    def grad_sync_deferred(self) -> bool: ...

    # --- training step ---
    def zero_grad(self) -> None: ...
    def set_grad_sync(self, enable: bool) -> None: ...
    def optimizer_step(self, *, max_grad_norm: float) -> float: ...
    def on_rollout_end(self) -> None: ...

    # --- eval-EMA swap ---
    def apply_eval_ema(self) -> None: ...
    def restore_from_eval(self) -> None: ...

    # --- checkpoint ---
    def save(self, path: str, step: Optional[int] = None, mode: str = "full") -> None: ...
    def load(self, path: str) -> int: ...

    # --- memory lifecycle ---
    def onload(self) -> None: ...
    def offload(self) -> None: ...

    # --- accessors ---
    def trainable_module(self) -> nn.Module: ...


__all__ = ["TrainBackend"]
