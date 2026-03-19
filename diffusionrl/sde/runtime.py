"""Shared SDE runtime entrypoints."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from diffusionrl.sde.kernels import get_sde_strategy


def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """Apply SD3-style time shift to timesteps in ``[0, 1]``."""

    return (shift * t) / (1 + (shift - 1) * t)


def get_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get the sigma (noise level) schedule."""

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    sigmas = sd3_time_shift(shift, timesteps)
    if device is not None:
        sigmas = sigmas.to(device)
    return sigmas


def compute_sde_log_prob(
    noise_pred: torch.Tensor,
    sample: torch.Tensor,
    prev_sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 1.0,
    sde_type: str = "flow",
    sigma_max: Optional[float] = None,
    use_sde_solver: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample log probability for a single SDE transition."""

    noise_pred = noise_pred.float()
    sample = sample.float()
    prev_sample = prev_sample.float()

    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    if sigma_next.dim() == 0:
        sigma_next = sigma_next.unsqueeze(0)

    while sigma.dim() < sample.dim():
        sigma = sigma.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)

    strategy = get_sde_strategy(sde_type)
    log_prob, prev_sample_mean = strategy.compute_log_prob(
        noise_pred=noise_pred,
        sample=sample,
        prev_sample=prev_sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=eta,
        sigma_max=sigma_max,
        use_sde_solver=use_sde_solver,
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return log_prob, prev_sample_mean


def sde_step_with_log_prob(
    noise_pred: torch.Tensor,
    sample: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    eta: float = 1.0,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: str = "flow",
    output_dtype: Optional[torch.dtype] = None,
    use_sde_solver: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one SDE sampling step and compute its log probability."""

    noise_pred = noise_pred.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    device = noise_pred.device
    sigma = sigmas[step_index].to(device)
    sigma_next = sigmas[step_index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_next - sigma

    strategy = get_sde_strategy(sde_type)
    prev_sample, log_prob, prev_sample_mean = strategy.step_with_log_prob(
        noise_pred=noise_pred,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        dt=dt,
        eta=eta,
        prev_sample=prev_sample,
        generator=generator,
        sigma_max=sigma_max,
        output_dtype=output_dtype,
        use_sde_solver=use_sde_solver,
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean


__all__ = [
    "sd3_time_shift",
    "get_sigma_schedule",
    "compute_sde_log_prob",
    "sde_step_with_log_prob",
]
