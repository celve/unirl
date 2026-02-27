"""
EMA (Exponential Moving Average) module for model parameters.

Used by NFT algorithm for dual adapter mechanism:
- new_adapter: Current policy being trained
- old_adapter: Reference policy (EMA of new_adapter)

Based on: DiffusionNFT/flow_grpo/ema.py
"""

from collections.abc import Iterable
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn as nn


class EMAModuleWrapper:
    """
    EMA wrapper for model parameters.

    Maintains an exponential moving average of model parameters,
    useful for:
    - Stable policy reference in RL (NFT old adapter)
    - Model averaging for better generalization
    - Rollout policy snapshots

    Args:
        parameters: Iterable of model parameters to track
        decay: EMA decay rate (higher = slower update, default 0.9999)
        update_step_interval: How often to update EMA (default 1)
        device: Device for EMA parameters

    Example:
        # Initialize EMA
        ema = EMAModuleWrapper(model.parameters(), decay=0.9999)

        # Training loop
        for step in range(num_steps):
            loss.backward()
            optimizer.step()
            ema.step(model.parameters(), step)

        # Use EMA weights for evaluation
        ema.copy_ema_to(model.parameters())
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: Optional[torch.device] = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters: Optional[List[torch.Tensor]] = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        """
        Get current decay rate with warmup.

        Uses warmup formula: min((1 + step) / (10 + step), decay)
        This starts with low decay and ramps up to target decay.

        Args:
            optimization_step: Current optimization step

        Returns:
            Current decay rate
        """
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(
        self,
        parameters: Iterable[nn.Parameter],
        optimization_step: int,
    ) -> None:
        """
        Update EMA parameters.

        Formula: ema = decay * ema + (1 - decay) * param

        Args:
            parameters: Current model parameters
            optimization_step: Current optimization step
        """
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        # In-place update when on same device
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # Memory-efficient cross-device update
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Move EMA parameters to device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(
        self,
        parameters: Iterable[nn.Parameter],
        store_temp: bool = True,
        grad: bool = False,
    ) -> None:
        """
        Copy EMA parameters to model parameters.

        Args:
            parameters: Model parameters to copy to
            store_temp: Whether to store original parameters temporarily
            grad: Whether to keep gradient info in temp storage
        """
        parameters = list(parameters)

        if store_temp:
            if grad:
                self.temp_stored_parameters = [p.data.clone() for p in parameters]
            else:
                self.temp_stored_parameters = [p.detach().cpu() for p in parameters]

        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    @torch.no_grad()
    def update(
        self,
        parameters: Optional[Iterable[nn.Parameter]] = None,
        optimization_step: int = 0,
    ) -> None:
        """
        Convenience method for updating EMA.

        If parameters not provided and model was passed to __init__, uses model.parameters().

        Args:
            parameters: Model parameters (optional if model was stored)
            optimization_step: Current optimization step (default 0)
        """
        if parameters is None:
            if hasattr(self, '_model') and self._model is not None:
                parameters = self._model.parameters()
            else:
                raise ValueError("parameters must be provided or model must be stored in __init__")
        self.step(parameters, optimization_step)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", self.ema_parameters)
        self.to(self.device)


class DualAdapterEMA:
    """
    EMA updater for dual LoRA adapter setup (NFT algorithm).

    NFT uses two LoRA adapters:
    - new_adapter (default): Current policy being trained
    - old_adapter: Reference policy updated via EMA

    The loss computation uses both:
    - positive_pred = beta * new + (1-beta) * old
    - negative_pred = (1+beta) * old - beta * new

    Args:
        decay: EMA decay rate for old adapter update
        old_adapter_name: Name of the old (reference) adapter
        new_adapter_name: Name of the new (training) adapter
        decay_fn: Optional callable(step) -> float for dynamic decay
                  If provided, overrides the fixed decay value.
                  Use this for DiffusionNFT-style dynamic decay.
    """

    def __init__(
        self,
        decay: float = 0.001,
        old_adapter_name: str = "old",
        new_adapter_name: str = "default",
        decay_fn: Optional[Callable[[int], float]] = None,
        # Optional schedule parameters (ignored if decay_fn is provided)
        decay_type: str = "constant",
        **kwargs,  # Absorb any unused params
    ):
        self.decay = decay
        self.old_adapter_name = old_adapter_name
        self.new_adapter_name = new_adapter_name
        self._step = 0

        # Set up decay function
        if decay_fn is not None:
            self._get_decay = decay_fn
        elif decay_type != "constant":
            # Build decay function from schedule params.
            self._get_decay = self._build_decay_fn(decay_type, kwargs)
        else:
            self._get_decay = None  # Use fixed decay

    def _build_decay_fn(self, decay_type: str, kwargs: dict) -> Callable[[int], float]:
        """Build decay function from schedule parameters."""
        flat_steps = kwargs.get("flat_steps", 0)
        uprate = kwargs.get("uprate", kwargs.get("ema_uprate", 0.001))
        uphold = kwargs.get("uphold", kwargs.get("ema_uphold", 0.5))

        if decay_type == "linear":
            return lambda step: min(step * uprate, uphold)
        elif decay_type == "warmup":
            return lambda step: 0.0 if step < flat_steps else min((step - flat_steps) * uprate, uphold)
        else:
            return lambda step: self.decay

    def get_decay(self, step: Optional[int] = None) -> float:
        """Get decay value for given step."""
        if self._get_decay is None:
            return self.decay
        return self._get_decay(step if step is not None else self._step)

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """
        Update old adapter weights using EMA from new adapter.

        Formula: old = decay * old + (1 - decay) * new

        Args:
            model: Model with dual LoRA adapters
            step: Optional step number for dynamic decay (uses internal counter if None)

        Returns:
            True if update was successful, False otherwise
        """
        # Handle FSDP-wrapped models
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            return False

        # Get decay for current step (fixed or dynamic)
        current_decay = self.get_decay(step)

        try:
            # Get new adapter parameters
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                name: param.data.clone()
                for name, param in adapter_model.named_parameters()
                if "lora" in name.lower()
            }

            # Update old adapter with EMA
            adapter_model.set_adapter(self.old_adapter_name)
            for name, param in adapter_model.named_parameters():
                if "lora" in name.lower() and name in new_params:
                    # old = decay * old + (1 - decay) * new
                    param.data = current_decay * param.data + (1 - current_decay) * new_params[name]

            # Switch back to new adapter for training
            adapter_model.set_adapter(self.new_adapter_name)

            # Increment internal step counter
            self._step += 1

            return True

        except Exception:
            return False
