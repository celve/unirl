"""Shared SDE runtime entrypoints."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from diffusionrl.sde.kernels import SDEStrategy
from diffusionrl.sde.registry import resolve_sde_strategy_class

if TYPE_CHECKING:
    from diffusionrl.sde.kernels import StepStrategy


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


def get_sigma_schedule_diffusers(
    num_steps: int,
    shift: float = 3.0,
    num_train_timesteps: int = 1000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get sigma schedule aligned with diffusers FlowMatchEulerDiscreteScheduler."""

    timesteps = torch.linspace(float(num_train_timesteps), 0.0, num_steps)
    sigmas = timesteps / num_train_timesteps
    sigmas = sd3_time_shift(shift, sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    if device is not None:
        sigmas = sigmas.to(device)
    return sigmas


# ---------------------------------------------------------------------------
# Unified SDE step entrypoint (new primary API)
# ---------------------------------------------------------------------------

def denoising_step(
    noise_pred: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 1.0,
    prev_sample: Optional[torch.Tensor] = None,
    sde_type: str = "flow",
    sigma_max: float = 0.99,
    strategy: Optional["StepStrategy"] = None,
    step_index: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unified SDE step: handles both sampling and training replay.

    This is the **single entrypoint** for all SDE / ODE transitions.

    * **Sampling** (``prev_sample=None``): generates noise and returns the
      new sample together with its elementwise log probability.
    * **Training replay** (``prev_sample`` provided): computes log
      probability of the given transition without generating noise.

    SDE step noise is always uncontrolled (no seeded generator).  Initial
    noise determinism is handled upstream by ``generate_latents`` via
    per-sample ``noise_group_ids`` + ``base_seed``.

    Args:
        noise_pred: Model velocity / noise prediction.
        sample: Current noisy sample x_t.
        sigma: Current noise level (scalar or broadcastable).
        sigma_next: Next noise level.
        eta: Stochasticity level.  ``0`` yields a deterministic Euler step.
        prev_sample: If provided, used for training-time log-prob replay
            instead of generating a new sample.
        sde_type: SDE formulation name (``flow``, ``cps``, ``dance``,
            ``dpm2``, …).
        sigma_max: Maximum sigma, used by some formulations for numerical
            stability at the boundary.
        strategy: Pre-built strategy instance.  When ``None``, a fresh
            instance is created from *sde_type*.  Pass an existing instance
            for stateful strategies (e.g. DPM2) that must persist across
            steps.
        step_index: Current step index (used by stateful strategies like
            DPM2).

    Returns:
        ``(prev_sample_out, log_prob, prev_sample_mean)``

        *log_prob* and *prev_sample_mean* are ``None`` when the step is
        deterministic (eta=0 or DPM2).  *log_prob* is per-sample (mean
        over spatial / channel dims).
    """
    _input_dtype = sample.dtype
    noise_pred = noise_pred.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    if sigma_next.dim() == 0:
        sigma_next = sigma_next.unsqueeze(0)

    while sigma.dim() < sample.dim():
        sigma = sigma.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)

    if strategy is None:
        strategy = resolve_sde_strategy_class(sde_type)()

    prev_sample, prev_sample_mean, std_var = strategy.step(
        noise_pred=noise_pred,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=eta,
        prev_sample=prev_sample,
        generator=None, # DONOT PASS GENERATOR HERE - It will hurt diversity and performance
        sigma_max=float(sigma_max),
        step_index=step_index,
    )

    if isinstance(strategy, SDEStrategy):
        if eta < 1e-7:
            log_prob = None
        else:
            prev_sample = prev_sample.to(dtype=_input_dtype).float()
            log_prob = strategy.compute_log_prob(
                prev_sample=prev_sample,
                prev_sample_mean=prev_sample_mean,
                std_var=std_var,
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = None

    return prev_sample, log_prob, prev_sample_mean

__all__ = [
    "sd3_time_shift",
    "get_sigma_schedule",
    "get_sigma_schedule_diffusers",
    "denoising_step",
]
