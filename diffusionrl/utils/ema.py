"""
EMA (Exponential Moving Average) module for model parameters.

Two use cases:
1. Eval EMA: all algorithms maintain a smoothed copy of trainable parameters
   for stable evaluation (decay=0.9, warmup formula).
2. NFT old-model EMA: NFT full-param training uses a second EMA instance
   (with its own decay schedule) as the reference / "old" policy.

For NFT LoRA training, the existing DualAdapterEMA handles old/new adapter
synchronization without touching this class.

FSDP/DTensor: When model parameters are DTensors (e.g. under FSDP2), all
EMA buffers and copies must stay DTensor; mixing torch.Tensor and DTensor
in copy_/in-place ops raises RuntimeError. This module keeps types consistent
and uses a type-aware copy when needed.
"""

from collections.abc import Iterable
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn


def _is_dtensor(t: torch.Tensor) -> bool:
    """True if t is a distributed tensor (e.g. FSDP sharded)."""
    return type(t).__name__ == "DTensor"


def _copy_into_param(param: nn.Parameter, src: torch.Tensor) -> None:
    """Copy src into param.data, handling Tensor vs DTensor so distributed ops see consistent types."""
    dest = param.data
    if type(dest) == type(src):
        dest.copy_(src)
        return
    if _is_dtensor(dest) and not _is_dtensor(src):
        from torch.distributed.tensor import distribute_tensor

        src_dt = distribute_tensor(src, dest.device_mesh, dest.placements)
        dest.copy_(src_dt)
        return
    if not _is_dtensor(dest) and _is_dtensor(src):
        full = src.full_tensor() if hasattr(src, "full_tensor") else src
        dest.copy_(full)
        return
    dest.copy_(src)


def _copy_from_param(dest: torch.Tensor, param: nn.Parameter) -> None:
    """Copy param.data into dest (e.g. EMA buffer), handling Tensor vs DTensor."""
    src = param.data.detach()
    if type(dest) == type(src):
        dest.copy_(src)
        return
    if _is_dtensor(dest) and not _is_dtensor(src):
        from torch.distributed.tensor import distribute_tensor

        src_dt = distribute_tensor(src, dest.device_mesh, dest.placements)
        dest.copy_(src_dt)
        return
    if not _is_dtensor(dest) and _is_dtensor(src):
        full = src.full_tensor() if hasattr(src, "full_tensor") else src
        dest.copy_(full)
        return
    dest.copy_(src)


class EMAModuleWrapper:
    """Exponential moving average of model parameters.

    Args:
        parameters: Iterable of parameters to track.
        decay: Target EMA decay rate.
        update_step_interval: Update EMA every N optimizer steps.
        device: Storage device for EMA tensors (``None`` keeps source device).
        decay_fn: Optional ``(step) -> float`` that overrides the default
            warmup-based decay schedule.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: Optional[torch.device] = None,
        decay_fn: Optional[Callable[[int], float]] = None,
    ):
        parameters = list(parameters)
        # When device is None (FSDP/sharded), do not call .to(device) so clone stays DTensor.
        if device is not None:
            self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        else:
            self.ema_parameters = [p.clone().detach() for p in parameters]
        self.temp_stored_parameters: Optional[list[torch.Tensor]] = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self._decay_fn = decay_fn
        self._step_counter = 0

    def get_current_decay(self, optimization_step: int) -> float:
        if self._decay_fn is not None:
            return self._decay_fn(optimization_step)
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[nn.Parameter], optimization_step: Optional[int] = None):
        """Update EMA: ``ema = decay * ema + (1 - decay) * param``."""
        if optimization_step is None:
            optimization_step = self._step_counter

        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

        self._step_counter += 1

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.device = device
        if device is None and dtype is None:
            return
        if device is not None:
            self.ema_parameters = [
                p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
                for p in self.ema_parameters
            ]
        elif dtype is not None:
            self.ema_parameters = [
                p.to(dtype=dtype) if p.is_floating_point() else p
                for p in self.ema_parameters
            ]

    @torch.no_grad()
    def sync_with_model(self, parameters: Iterable[nn.Parameter]) -> None:
        """Force EMA to be an exact copy of current model parameters."""
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            _copy_from_param(ema_parameter, parameter)

    def copy_ema_to(
        self,
        parameters: Iterable[nn.Parameter],
        store_temp: bool = True,
        grad: bool = False,
    ) -> None:
        """Copy EMA weights into model parameters.

        Args:
            parameters: Model parameters to overwrite.
            store_temp: If ``True``, save originals so they can be restored
                via :meth:`copy_temp_to`.
            grad: When *store_temp* is ``True``, keep temp tensors on the
                same device (``grad=True``) or move to CPU (``grad=False``).
                When ``device is None`` (FSDP / sharded), temps always stay
                on-device to preserve distributed tensor metadata.
        """
        parameters = list(parameters)
        if store_temp:
            if grad or self.device is None:
                self.temp_stored_parameters = [p.data.clone() for p in parameters]
            else:
                self.temp_stored_parameters = [p.detach().cpu() for p in parameters]

        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            _copy_into_param(parameter, ema_parameter)

    def copy_temp_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """Restore model parameters previously saved by *copy_ema_to*."""
        assert self.temp_stored_parameters is not None, "No temp parameters stored"
        parameters = list(parameters)
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            _copy_into_param(parameter, temp_parameter)
        self.temp_stored_parameters = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", self.ema_parameters)
        self._step_counter = int(state_dict.get("step_counter", 0))
        self.to(self.device)


class DualAdapterEMA:
    """EMA updater for dual LoRA adapter setup (NFT algorithm).

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
        decay_type: str = "constant",
        **kwargs,
    ):
        self.decay = decay
        self.old_adapter_name = old_adapter_name
        self.new_adapter_name = new_adapter_name
        self._step = 0

        if decay_fn is not None:
            self._get_decay = decay_fn
        elif decay_type != "constant":
            self._get_decay = self._build_decay_fn(decay_type, kwargs)
        else:
            self._get_decay = None

    def _build_decay_fn(self, decay_type: str, kwargs: dict) -> Callable[[int], float]:
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
        if self._get_decay is None:
            return self.decay
        return self._get_decay(step if step is not None else self._step)

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """Update old adapter weights using EMA from new adapter.

        Formula: old = decay * old + (1 - decay) * new
        """
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            return False

        current_decay = self.get_decay(step)

        try:
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                name: param.data.clone()
                for name, param in adapter_model.named_parameters()
                if "lora" in name.lower()
            }

            adapter_model.set_adapter(self.old_adapter_name)
            for name, param in adapter_model.named_parameters():
                if "lora" in name.lower() and name in new_params:
                    param.data = current_decay * param.data + (1 - current_decay) * new_params[name]

            adapter_model.set_adapter(self.new_adapter_name)
            self._step += 1
            return True

        except Exception:
            return False
