"""SDE kernel registry for shared transition math."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import torch

from diffusionrl.sde.rules import normalize_sde_type


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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute elementwise log probability and transition mean."""

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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a sampling step and return sample, elementwise log prob, mean."""


SDE_STRATEGY_REGISTRY: Dict[str, Type[SDEStrategy]] = {}


def register_sde_strategy(*names: str):
    """Decorator to register an SDE strategy under one or more names."""

    def decorator(cls: Type[SDEStrategy]) -> Type[SDEStrategy]:
        for name in names:
            SDE_STRATEGY_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_sde_strategy(sde_type: str) -> SDEStrategy:
    """Get a strategy instance by SDE type name.

    Note: "dpm2" is declared in CANONICAL_SDE_TYPES (rules.py) but has no
    registered SDEStrategy — it is a deterministic ODE solver that bypasses
    the stochastic sampling path entirely.  Calling this function with
    sde_type="dpm2" will raise ValueError.  Callers should check
    ``is_deterministic_sde_type()`` before invoking this function.
    """

    normalized = normalize_sde_type(sde_type)
    cls = SDE_STRATEGY_REGISTRY.get(normalized)
    if cls is None:
        raise ValueError(
            f"Unknown sde_type: {sde_type!r} (normalized={normalized!r}). "
            f"Available: {sorted(SDE_STRATEGY_REGISTRY.keys())}"
        )
    return cls()


@register_sde_strategy("flow")
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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sigma_max is None:
            sigma_max = 1.0
        dt = sigma_next - sigma
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        std_dev_t = torch.sqrt(
            sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
        ) * eta

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

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
    """Coefficient-preserving sampling."""

    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = (
            pred_original * (1 - sigma_next)
            + noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)
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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = (
            pred_original * (1 - sigma_next)
            + noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)
        )

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype
            )
            prev_sample = prev_sample_mean + std_dev_t * noise

        if output_dtype is not None and output_dtype != prev_sample.dtype:
            prev_sample = prev_sample.to(dtype=output_dtype)

        log_prob = -((prev_sample.detach().float() - prev_sample_mean) ** 2)
        return prev_sample, log_prob, prev_sample_mean


@register_sde_strategy("dance")
class DanceSDEStrategy(SDEStrategy):
    """DanceGRPO SDE formulation."""

    def compute_log_prob(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        sigma_max: Optional[float] = None,
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dsigma = sigma_next - sigma
        delta_t = sigma - sigma_next
        std_dev_t = eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * noise_pred
        prev_sample_mean = sample + dsigma * noise_pred

        if use_sde_solver:
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
        use_sde_solver: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        dsigma = sigma_next - sigma
        delta_t = sigma - sigma_next
        std_dev_t = eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * noise_pred
        prev_sample_mean = sample + dsigma * noise_pred

        if use_sde_solver:
            score_estimate = -(sample - pred_original * (1 - sigma)) / sigma**2
            log_term = -0.5 * (eta**2) * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        if prev_sample is None:
            noise = randn_tensor(
                noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype
            )
            prev_sample = prev_sample_mean + std_dev_t * noise

        if output_dtype is not None and output_dtype != prev_sample.dtype:
            prev_sample = prev_sample.to(dtype=output_dtype)

        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean) ** 2) / (2 * std_dev_t**2)
            - torch.log(std_dev_t)
            - 0.5 * math.log(2 * math.pi)
        )
        return prev_sample, log_prob, prev_sample_mean


__all__ = [
    "SDEStrategy",
    "SDE_STRATEGY_REGISTRY",
    "register_sde_strategy",
    "get_sde_strategy",
    "FlowSDEStrategy",
    "CPSSDEStrategy",
    "DanceSDEStrategy",
]
