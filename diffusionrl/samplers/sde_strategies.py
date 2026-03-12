"""SDE strategy registry for log probability computation.

Each strategy encapsulates the SDE-specific math for:
- compute_log_prob: log probability of a transition given (noise_pred, sample, prev_sample)
- step_with_log_prob: sampling step + log probability computation

New SDE types can be added by registering a strategy class.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import torch


class SDEStrategy(ABC):
    """Base class for SDE log probability computation strategies."""

    @abstractmethod
    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability and SDE mean for one step.

        Args:
            noise_pred: Model velocity prediction [B, C, ...]
            sample: Current sample x_t
            prev_sample: Next sample x_{t-1}
            sigma: Current sigma (broadcastable to sample shape)
            sigma_next: Next sigma
            eta: Noise level
            sigma_max: Maximum sigma value

        Returns:
            log_prob: [B] per-sample log probability (before batch-mean reduction)
            prev_sample_mean: [B, C, ...] SDE mean
        """
        ...

    @abstractmethod
    def step_with_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        dt: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 1.0,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampling step with log probability.

        Args:
            noise_pred: Model velocity prediction
            sample: Current noisy sample x_t
            sigma: Current sigma
            sigma_next: Next sigma
            dt: sigma_next - sigma
            eta: Noise level
            prev_sample: Pre-computed previous sample (for training)
            generator: Random number generator
            sigma_max: Maximum sigma
            output_dtype: If set, cast prev_sample to this dtype before
                computing log_prob, so the log-prob is evaluated on the
                exact precision that will be stored in the trajectory.

        Returns:
            prev_sample: Next sample x_{t-1} (in output_dtype if specified)
            log_prob: [B, C, ...] element-wise log probability (before spatial mean)
            prev_sample_mean: [B, C, ...] SDE mean
        """
        ...


# Registry
SDE_STRATEGY_REGISTRY: Dict[str, Type[SDEStrategy]] = {}


def register_sde_strategy(*names: str):
    """Decorator to register an SDE strategy under one or more names."""
    def decorator(cls: Type[SDEStrategy]) -> Type[SDEStrategy]:
        for name in names:
            SDE_STRATEGY_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_sde_strategy(sde_type: str) -> SDEStrategy:
    """Get a strategy instance by SDE type name."""
    cls = SDE_STRATEGY_REGISTRY.get(sde_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown sde_type: {sde_type!r}. "
            f"Available: {sorted(SDE_STRATEGY_REGISTRY.keys())}"
        )
    return cls()


# --- Concrete strategies ---


@register_sde_strategy("sde", "flux_flow", "flow")
class FlowSDEStrategy(SDEStrategy):
    """Standard SDE formulation from flow-GRPO."""

    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sigma_max is None:
            sigma_max = 1.0
        dt = sigma_next - sigma
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt) +
            noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        variance = (std_dev_t * torch.sqrt(-dt)) ** 2
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * variance)
            - torch.log(std_dev_t * torch.sqrt(-dt))
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob, prev_sample_mean

    def step_with_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        dt: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 1.0,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        std_dev_t = torch.sqrt(
            sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
        ) * eta

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt) +
            noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

        # Cast prev_sample to storage dtype before computing log_prob,
        # so old_log_prob (sampling) matches new_log_prob (training).
        if output_dtype is not None and output_dtype != prev_sample.dtype:
            prev_sample = prev_sample.to(dtype=output_dtype)

        variance = (std_dev_t * torch.sqrt(-dt)) ** 2
        std_term = std_dev_t * torch.sqrt(-dt)
        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean) ** 2) / (2 * variance)
            - torch.log(std_term)
            - 0.5 * math.log(2 * math.pi)
        )
        return prev_sample, log_prob, prev_sample_mean


@register_sde_strategy("cps")
class CPSSDEStrategy(SDEStrategy):
    """Coefficient-Preserving Sampling."""

    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = (
            pred_original * (1 - sigma_next) +
            noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)
        )
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        return log_prob, prev_sample_mean

    def step_with_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        dt: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 1.0,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = (
            pred_original * (1 - sigma_next) +
            noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)
        )

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * noise

        # Cast prev_sample to storage dtype before computing log_prob
        if output_dtype is not None and output_dtype != prev_sample.dtype:
            prev_sample = prev_sample.to(dtype=output_dtype)

        log_prob = -((prev_sample.detach().float() - prev_sample_mean) ** 2)
        return prev_sample, log_prob, prev_sample_mean


@register_sde_strategy("dance", "flux_dance")
class DanceSDEStrategy(SDEStrategy):
    """DanceGRPO SDE formulation (for Flux)."""

    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dsigma = sigma_next - sigma
        delta_t = sigma - sigma_next
        std_dev_t = eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * noise_pred
        prev_sample_mean = sample + dsigma * noise_pred

        score_estimate = -(sample - pred_original * (1 - sigma)) / (sigma**2)
        log_term = -0.5 * (eta**2) * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t**2)
            - torch.log(std_dev_t)
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob, prev_sample_mean

    def step_with_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        dt: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 1.0,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        dsigma = sigma_next - sigma
        delta_t = sigma - sigma_next
        std_dev_t = eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * noise_pred
        prev_sample_mean = sample + dsigma * noise_pred

        score_estimate = -(sample - pred_original * (1 - sigma)) / sigma**2
        log_term = -0.5 * (eta**2) * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * noise

        # Cast prev_sample to storage dtype before computing log_prob
        if output_dtype is not None and output_dtype != prev_sample.dtype:
            prev_sample = prev_sample.to(dtype=output_dtype)

        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean) ** 2) / (2 * std_dev_t**2)
            - torch.log(std_dev_t)
            - 0.5 * math.log(2 * math.pi)
        )
        return prev_sample, log_prob, prev_sample_mean
