"""Unit tests for ``FSDPBackend.optimizer_step``'s non-finite-grad guard.

``optimizer_step`` is the single optimizer-step chokepoint every v2 trainer
(PE / VLM / diffusion via ``TrainStack``) routes through; the guard skips the
whole step on a NaN/Inf clipped grad norm so a bad norm can't scale and poison
the weights. The method only touches ``model`` / ``optimizer`` / ``scheduler``
/ ``ema`` / ``_optimizer_step_count``, so we drive it on a tiny CPU model with
``object.__new__`` (bypassing the FSDP-wrapping ``__init__``) to exercise both
branches without FSDP / a process group.
"""

from __future__ import annotations

import math

import torch

from diffusionrl.train.backend.fsdp import FSDPBackend


def _make_backend(model: torch.nn.Module) -> FSDPBackend:
    backend = object.__new__(FSDPBackend)
    backend.model = model
    backend.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backend.scheduler = None
    backend.ema = None
    backend._optimizer_step_count = 0
    return backend


def test_optimizer_step_applies_finite_grad() -> None:
    model = torch.nn.Linear(2, 2)
    backend = _make_backend(model)
    before = model.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    grad_norm = backend.optimizer_step(max_grad_norm=1.0)

    assert math.isfinite(grad_norm)
    assert backend._optimizer_step_count == 1
    assert not torch.equal(model.weight.detach(), before)  # weights stepped


def test_optimizer_step_skips_non_finite_grad() -> None:
    model = torch.nn.Linear(2, 2)
    backend = _make_backend(model)
    before = model.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.weight.grad[0, 0] = float("nan")  # poison the global grad norm

    grad_norm = backend.optimizer_step(max_grad_norm=1.0)

    assert not math.isfinite(grad_norm)  # non-finite norm reported back for logging
    assert backend._optimizer_step_count == 0  # step skipped, count not advanced
    assert torch.equal(model.weight.detach(), before)  # weights untouched
    assert all(p.grad is None for p in model.parameters())  # grads cleared (set_to_none)
