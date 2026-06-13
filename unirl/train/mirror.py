"""Full-model mirror-parameter injection for EMA.

:func:`inject_mirror` registers a frozen ``shadow_*`` parameter beside every
trainable param and returns a :class:`~unirl.train.shadow.Shadow` handle the
full-model EMA consumes.  Stamps its post-materialize shadow copy via
``unirl.train.deferred``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import List, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter

from unirl.train.deferred import _stamp
from unirl.train.shadow import Shadow

logger = logging.getLogger(__name__)


def inject_mirror(
    model: nn.Module,
    *,
    prefix: str = "shadow_",
) -> Shadow:
    """Register shadow_* parameters for full-model EMA.  Returns Shadow."""
    pairs: List[Tuple[nn.Module, str, str]] = []

    for fqn, p in list(model.named_parameters()):
        if not p.requires_grad:
            continue
        parent, attr = _parent_and_attr(model, fqn)
        shadow_attr = prefix + attr
        shadow_param = Parameter(torch.empty_like(p), requires_grad=False)
        parent.register_parameter(shadow_attr, shadow_param)
        pairs.append((parent, attr, shadow_attr))

    if _current_rank() == 0:
        logger.info("inject_mirror: registered %d shadow parameters (prefix=%r)", len(pairs), prefix)

    _stamp(model, partial(_copy_mirror, pairs=pairs))

    return Shadow(
        iter_pairs=lambda: ((getattr(m, a), getattr(m, s)) for m, a, s in pairs),
        swap_in=lambda: _swap_mirror(pairs),
        swap_out=lambda: _swap_mirror(pairs),
    )


def _copy_mirror(model: nn.Module, *, pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        getattr(mod, shadow_attr).data.copy_(getattr(mod, live_attr).data)


def _swap_mirror(pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        live = getattr(mod, live_attr)
        shd = getattr(mod, shadow_attr)
        live.data, shd.data = shd.data, live.data


def _parent_and_attr(model: nn.Module, fqn: str) -> Tuple[nn.Module, str]:
    parts = fqn.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[0].split("."):
        parent = getattr(parent, part)
    return parent, parts[1]


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["inject_mirror"]
