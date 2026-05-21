"""Exponential moving average of model parameters.

Used by :class:`diffusionrl.training.ema_policy.EMAPolicy` to track
a shadow copy of the trainable parameters for eval. NFT's dual-adapter
reference EMA lives directly on
:class:`diffusionrl.training.nft_lora_policy.NFTLoRAPolicy`; this
module is intentionally narrow to a single primitive.

FSDP / DTensor: when model parameters are DTensors (e.g. under FSDP2),
all EMA buffers, temp backups, and copy operations must stay as
DTensors. Mixing plain ``Tensor`` and ``DTensor`` in ``copy_`` triggers
a RuntimeError. The FSDP backend sets ``reshard_after_forward=True`` so
``model.parameters()`` returns DTensors consistently before and after
forward passes.
"""

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


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
        current_decay = self.get_current_decay(optimization_step)
        one_minus_decay = 1 - current_decay

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        # In-place: ema = decay * ema + (1 - decay) * param
                        ema_parameter.mul_(current_decay).add_(parameter, alpha=one_minus_decay)
                    else:
                        param_copy = parameter.detach().to(ema_parameter.device)
                        ema_parameter.mul_(current_decay).add_(param_copy, alpha=one_minus_decay)
                        del param_copy

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
            self.ema_parameters = [p.to(dtype=dtype) if p.is_floating_point() else p for p in self.ema_parameters]

    @torch.no_grad()
    def sync_with_model(self, parameters: Iterable[nn.Parameter]) -> None:
        """Force EMA to be an exact copy of current model parameters."""
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            ema_parameter.data.copy_(parameter.data)

    def copy_ema_to(
        self,
        parameters: Iterable[nn.Parameter],
        store_temp: bool = True,
    ) -> None:
        """Copy EMA weights into model parameters.

        Args:
            parameters: Model parameters to overwrite.
            store_temp: If ``True``, save originals so they can be restored
                via :meth:`copy_temp_to`.

        ``param.data.clone()`` preserves DTensor type, and
        ``param.data.copy_(ema)`` stays within DTensor dispatch. For
        non-FSDP (device!=None), temps go to CPU.
        """
        parameters = list(parameters)
        if store_temp:
            if self.device is None:
                # FSDP/DTensor path: clone preserves DTensor metadata.
                self.temp_stored_parameters = [p.data.clone() for p in parameters]
            else:
                # Non-FSDP path: materialize to CPU.
                self.temp_stored_parameters = [p.detach().cpu().clone() for p in parameters]

        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            if parameter.numel() > 0:
                # DTensor→DTensor or Tensor→Tensor: copy_ dispatch handles both.
                parameter.data.copy_(ema_parameter)

    def copy_temp_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """Restore model parameters previously saved by *copy_ema_to*."""
        assert self.temp_stored_parameters is not None, "No temp parameters stored"
        parameters = list(parameters)
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            # DTensor→DTensor or CPU Tensor→GPU Tensor: copy_ handles both.
            parameter.data.copy_(temp_parameter)
        self.temp_stored_parameters = None

    @contextmanager
    def use_ema_parameters(self, parameters: Iterable[nn.Parameter]):
        """Context manager: swap EMA weights in, automatically restore on exit."""
        parameters = list(parameters)
        self.copy_ema_to(parameters, store_temp=True)
        try:
            yield
        finally:
            self.copy_temp_to(parameters)

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


__all__ = ["EMAModuleWrapper"]
