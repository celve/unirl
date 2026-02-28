"""
Log probability computation for SDE steps.

This module contains the core formulas for computing log probabilities
during SDE sampling. SDE-type-specific math is delegated to the strategy
registry in ``sde_strategies.py``.
"""

import math
from typing import Tuple, Optional
import torch

from diffusionrl.samplers.sde_strategies import get_sde_strategy


def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """Apply time shift to timesteps.

    SD3 uses shift=3.0 by default.

    Args:
        shift: Shift parameter (higher = more focus on early steps)
        t: Timesteps in [0, 1]

    Returns:
        Shifted timesteps
    """
    return (shift * t) / (1 + (shift - 1) * t)


def get_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get the sigma (noise level) schedule.

    Args:
        num_steps: Number of denoising steps
        shift: Time shift parameter
        device: Device for the schedule

    Returns:
        Tensor of sigmas [num_steps + 1]
    """
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
    sde_type: str = "sde",
    sigma_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probability for SDE step.

    This is the core formula for GRPO:
        x_{t-1} ~ N(mean, variance)
        log_prob = log N(x_{t-1} | mean, variance)
                 = -||x_{t-1} - mean||² / (2σ²) - log(σ) - 0.5*log(2π)

    The mean depends on the SDE type and the model's velocity prediction.

    Args:
        noise_pred: Model velocity prediction [B, C, H, W] or [B, C, T, H, W]
        sample: Current sample x_t
        prev_sample: Next sample x_{t-1}
        sigma: Current sigma (noise level)
        sigma_next: Next sigma
        eta: Noise level (controls stochasticity)
        sde_type: "sde" | "cps" | "dance" | "flux_dance" | "flux_flow"
        sigma_max: Maximum sigma value (defaults to sigma[1] or 1.0)

    Returns:
        log_prob: [B] per-sample log probability
        prev_sample_mean: [B, C, ...] SDE mean (for KL computation)
    """
    # Convert to float32 for numerical stability
    noise_pred = noise_pred.float()
    sample = sample.float()
    prev_sample = prev_sample.float()

    # Handle sigma shapes
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    if sigma_next.dim() == 0:
        sigma_next = sigma_next.unsqueeze(0)

    # Expand sigma for broadcasting
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
    )

    # Mean along all but batch dimension to get per-sample log prob
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
    sde_type: str = "sde",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SDE sampling step with log probability computation.

    This is the main function used during sampling to:
    1. Generate the next sample (if prev_sample is None)
    2. Compute the log probability of the transition

    Args:
        noise_pred: Model velocity prediction
        sample: Current noisy sample x_t
        sigmas: Sigma schedule [num_steps+1]
        step_index: Current step index
        eta: Noise level
        prev_sample: Pre-computed previous sample (for training)
        generator: Random number generator
        sde_type: SDE formulation

    Returns:
        prev_sample: Next sample x_{t-1}
        log_prob: Log probability [B]
        prev_sample_mean: SDE mean [B, C, ...]
    """
    # Convert to float32
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
    )

    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean


def flux_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    eta: float = 1.0,
    use_sde_solver: bool = False,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DanceGRPO-style SDE step with log probability for Flux/HunyuanVideo.

    This is the shared implementation used by both flux_sampler and hunyuan_sampler.
    Aligned with DanceGRPO flux_step formulation.

    Args:
        model_output: Velocity prediction from transformer
        latents: Current noisy latents x_t
        sigmas: Sigma schedule [num_steps + 1]
        index: Current step index
        eta: Noise level (controls stochasticity)
        use_sde_solver: Whether to apply SDE solver correction
        prev_sample: Pre-computed previous sample (for training log-prob recomputation)

    Returns:
        prev_sample: Next sample x_{t-1}
        pred_original_sample: Predicted clean sample x_0
        log_prob: Log probability [B]
    """
    # Keep all tensors on model_output device (critical for FSDP CPU offload path).
    target_device = model_output.device
    latents = latents.to(device=target_device)
    if sigmas.device != target_device:
        sigmas = sigmas.to(device=target_device)
    if prev_sample is not None and prev_sample.device != target_device:
        prev_sample = prev_sample.to(device=target_device)

    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma

    # ODE mean
    prev_sample_mean = latents + dsigma * model_output

    # Predicted clean sample
    pred_original_sample = latents - sigma * model_output

    # SDE noise std
    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t.item())

    # SDE solver correction
    if use_sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2 + 1e-12)
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    # Sample if not provided
    if prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # Log probability: log N(prev_sample | prev_sample_mean, std_dev_t²)
    log_prob = (
        -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
        / (2 * (std_dev_t**2))
        - math.log(std_dev_t)
        - torch.log(
            torch.sqrt(
                torch.tensor(2 * math.pi, device=prev_sample_mean.device, dtype=torch.float32)
            )
        )
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_original_sample, log_prob
