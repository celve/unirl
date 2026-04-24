"""Factory functions that turn typed training sub-configs into runtime torch objects.

Kept separate from ``diffusionrl/config/training_sections.py`` so the config layer
stays torch-free — launcher scripts, linters, and schema tools can import the
typed dataclasses without pulling ``torch``.

Backend override dispatch lives here: callers do not have to re-implement
the "try ``backend.build_*`` first, fall back to the default construction"
pattern at every call site. The new-style hooks on
``diffusionrl.training.backends.TrainBackend`` consume the typed
dataclasses directly, so no ``asdict`` roundtrip is needed.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from diffusionrl.config.training_sections import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from diffusionrl.training.backends.protocols import (
    LRSchedulerProtocol,
    OptimizerProtocol,
)


def build_optimizer(
    config: OptimizerConfig,
    *,
    params: Iterable[torch.nn.Parameter],
    backend: Any = None,
    actor: Any = None,
) -> OptimizerProtocol:
    """Build an optimizer from a typed :class:`OptimizerConfig`.

    If ``backend`` is provided and its ``build_optimizer`` hook returns a
    non-None value, that takes precedence. Otherwise the default
    ``torch.optim.AdamW`` construction is used.

    ``params`` is consulted only on the default path; backend overrides
    are expected to pull parameters from their own model reference.
    """
    del actor
    if backend is not None:
        backend_optimizer = backend.build_optimizer(config)
        if backend_optimizer is not None:
            return backend_optimizer

    trainable = [p for p in params if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
        weight_decay=float(config.weight_decay),
    )


def build_lr_scheduler(
    config: LrSchedulerConfig,
    *,
    optimizer: OptimizerProtocol,
    backend: Any = None,
    actor: Any = None,
) -> Optional[LRSchedulerProtocol]:
    """Build an LR scheduler from a typed :class:`LrSchedulerConfig`.

    Supports the same backend-override path as :func:`build_optimizer`.
    Returns ``None`` if ``config.type`` is not one of the supported values
    (``constant`` / ``linear`` / ``cosine``) and the backend did not provide
    an override.
    """
    del actor
    if backend is not None:
        backend_scheduler = backend.build_scheduler(config, optimizer)
        if backend_scheduler is not None:
            return backend_scheduler

    scheduler_type = str(config.type)
    warmup_steps = int(config.warmup_steps)
    total_steps = int(config.total_steps)

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    if scheduler_type == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=0,
        )

    return None


__all__ = ["build_lr_scheduler", "build_optimizer"]
