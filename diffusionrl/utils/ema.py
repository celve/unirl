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
                self.temp_stored_parameters = [p.detach().cpu().clone() for p in parameters]

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

    @staticmethod
    def _adapter_param_key(name: str, adapter_name: str) -> Optional[str]:
        token = f".{adapter_name}."
        if token not in name:
            return None
        return name.replace(token, ".__adapter__.")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": float(self.decay),
            "old_adapter_name": str(self.old_adapter_name),
            "new_adapter_name": str(self.new_adapter_name),
            "step": int(self._step),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.old_adapter_name = str(state_dict.get("old_adapter_name", self.old_adapter_name))
        self.new_adapter_name = str(state_dict.get("new_adapter_name", self.new_adapter_name))
        self._step = int(state_dict.get("step", self._step))

    @torch.no_grad()
    def sync_from_new(self, model: nn.Module) -> bool:
        """Force old adapter to exactly match new adapter or raise on failure."""
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            raise RuntimeError(
                "NFT adapter EMA requires a model exposing set_adapter(). "
                f"Got model type={type(adapter_model).__name__}."
            )

        try:
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                key: param.data.clone()
                for name, param in adapter_model.named_parameters()
                for key in [self._adapter_param_key(name, self.new_adapter_name)]
                if key is not None
            }
            if not new_params:
                raise RuntimeError(
                    "NFT adapter EMA could not find any parameters for the new adapter "
                    f"{self.new_adapter_name!r}."
                )

            adapter_model.set_adapter(self.old_adapter_name)
            copied = 0
            for name, param in adapter_model.named_parameters():
                key = self._adapter_param_key(name, self.old_adapter_name)
                if key is None or key not in new_params:
                    continue
                param.data.copy_(new_params[key])
                copied += 1
            if copied == 0:
                raise RuntimeError(
                    "NFT adapter EMA could not find any parameters for the old adapter "
                    f"{self.old_adapter_name!r}."
                )

            adapter_model.set_adapter(self.new_adapter_name)
            return True
        except Exception as exc:
            try:
                adapter_model.set_adapter(self.new_adapter_name)
            except Exception:
                pass
            raise RuntimeError(
                "Failed to initialize NFT old adapter from the new adapter. "
                f"new_adapter={self.new_adapter_name!r}, old_adapter={self.old_adapter_name!r}."
            ) from exc

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """Update old adapter weights using EMA from new adapter or raise on failure.

        Formula: old = decay * old + (1 - decay) * new
        """
        adapter_model = model.module if hasattr(model, "module") else model

        if not hasattr(adapter_model, "set_adapter"):
            raise RuntimeError(
                "NFT adapter EMA requires a model exposing set_adapter(). "
                f"Got model type={type(adapter_model).__name__}."
            )

        current_decay = self.get_decay(step)

        try:
            adapter_model.set_adapter(self.new_adapter_name)
            new_params = {
                key: param.data.clone()
                for name, param in adapter_model.named_parameters()
                for key in [self._adapter_param_key(name, self.new_adapter_name)]
                if key is not None
            }
            if not new_params:
                raise RuntimeError(
                    "NFT adapter EMA could not find any parameters for the new adapter "
                    f"{self.new_adapter_name!r}."
                )

            adapter_model.set_adapter(self.old_adapter_name)
            updated = 0
            for name, param in adapter_model.named_parameters():
                key = self._adapter_param_key(name, self.old_adapter_name)
                if key is None or key not in new_params:
                    continue
                param.data.mul_(current_decay).add_(new_params[key], alpha=(1 - current_decay))
                updated += 1
            if updated == 0:
                raise RuntimeError(
                    "NFT adapter EMA could not find any parameters for the old adapter "
                    f"{self.old_adapter_name!r}."
                )

            adapter_model.set_adapter(self.new_adapter_name)
            self._step += 1
            return True

        except Exception as exc:
            try:
                adapter_model.set_adapter(self.new_adapter_name)
            except Exception:
                pass
            raise RuntimeError(
                "Failed to update NFT old adapter from the new adapter via EMA. "
                f"new_adapter={self.new_adapter_name!r}, old_adapter={self.old_adapter_name!r}, "
                f"step={step if step is not None else self._step}, decay={current_decay}."
            ) from exc


class EMAManager:
    """Runtime EMA mechanism materialized from algorithm-declared policy."""

    def __init__(
        self,
        *,
        eval_ema: Optional[EMAModuleWrapper] = None,
        reference_param_ema: Optional[EMAModuleWrapper] = None,
        reference_adapter_ema: Optional[DualAdapterEMA] = None,
        reference_update_timing: str = "optimizer_step",
    ) -> None:
        self.eval_ema = eval_ema
        self.reference_param_ema = reference_param_ema
        self.reference_adapter_ema = reference_adapter_ema
        self.reference_update_timing = str(reference_update_timing).strip().lower()
        self._optimizer_step = 0

    @classmethod
    def from_model_and_spec(
        cls,
        *,
        model: nn.Module,
        spec: Any,
        use_lora: bool,
        uses_sharded_model: bool,
        algorithm: Optional[Any] = None,
    ) -> "EMAManager":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            return cls()

        ema_device = None if uses_sharded_model else torch.device("cpu")
        eval_ema = None
        if bool(getattr(spec, "enable_eval_ema", True)):
            eval_ema = EMAModuleWrapper(
                trainable_params,
                decay=float(getattr(spec, "eval_decay", 0.9)),
                update_step_interval=max(1, int(getattr(spec, "eval_update_interval", 1))),
                device=ema_device,
            )

        reference_mode = str(getattr(spec, "reference_mode", "none") or "none").strip().lower()
        reference_param_ema = None
        reference_adapter_ema = None
        if reference_mode == "nft_old_policy":
            if use_lora:
                reference_adapter_ema = DualAdapterEMA(
                    decay=float(getattr(spec, "reference_decay", 0.001)),
                    decay_type=str(getattr(spec, "reference_decay_type", "constant")),
                    flat_steps=int(getattr(spec, "reference_flat_steps", 0)),
                    uprate=float(getattr(spec, "reference_uprate", 0.001)),
                    uphold=float(getattr(spec, "reference_uphold", 0.5)),
                    old_adapter_name=str(getattr(spec, "old_adapter_name", "old")),
                    new_adapter_name=str(getattr(spec, "new_adapter_name", "default")),
                )
            else:
                reference_param_ema = EMAModuleWrapper(
                    trainable_params,
                    decay=float(getattr(spec, "reference_decay", 0.001)),
                    decay_fn=cls._build_reference_decay_fn(spec),
                    update_step_interval=1,
                    device=ema_device,
                )

        manager = cls(
            eval_ema=eval_ema,
            reference_param_ema=reference_param_ema,
            reference_adapter_ema=reference_adapter_ema,
            reference_update_timing=str(getattr(spec, "reference_update_timing", "optimizer_step")),
        )
        if reference_adapter_ema is not None:
            reference_adapter_ema.sync_from_new(model)
        manager.bind_algorithm(algorithm)
        return manager

    @staticmethod
    def _build_reference_decay_fn(spec: Any) -> Callable[[int], float]:
        decay_type = str(getattr(spec, "reference_decay_type", "constant"))
        base_decay = float(getattr(spec, "reference_decay", 0.001))
        flat_steps = int(getattr(spec, "reference_flat_steps", 0))
        uprate = float(getattr(spec, "reference_uprate", 0.001))
        uphold = float(getattr(spec, "reference_uphold", 0.5))
        if decay_type == "linear":
            return lambda step: min(step * uprate, uphold)
        if decay_type == "warmup":
            return lambda step: 0.0 if step < flat_steps else min((step - flat_steps) * uprate, uphold)
        return lambda _step: float(base_decay)

    def bind_algorithm(self, algorithm: Optional[Any]) -> None:
        if algorithm is None or self.reference_param_ema is None:
            return
        setattr(algorithm, "_old_params_ema", self.reference_param_ema)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "eval_ema": (
                self.eval_ema.state_dict()
                if self.eval_ema is not None
                else None
            ),
            "reference_param_ema": (
                self.reference_param_ema.state_dict()
                if self.reference_param_ema is not None
                else None
            ),
            "reference_adapter_ema": (
                self.reference_adapter_ema.state_dict()
                if self.reference_adapter_ema is not None
                else None
            ),
            "reference_update_timing": self.reference_update_timing,
            "optimizer_step": int(self._optimizer_step),
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        *,
        algorithm: Optional[Any] = None,
    ) -> None:
        eval_state = state_dict.get("eval_ema")
        if self.eval_ema is not None and isinstance(eval_state, dict):
            self.eval_ema.load_state_dict(eval_state)

        reference_param_state = state_dict.get("reference_param_ema")
        if self.reference_param_ema is not None and isinstance(reference_param_state, dict):
            self.reference_param_ema.load_state_dict(reference_param_state)

        reference_adapter_state = state_dict.get("reference_adapter_ema")
        if self.reference_adapter_ema is not None and isinstance(reference_adapter_state, dict):
            self.reference_adapter_ema.load_state_dict(reference_adapter_state)

        self.reference_update_timing = str(
            state_dict.get("reference_update_timing", self.reference_update_timing)
        ).strip().lower()
        self._optimizer_step = int(state_dict.get("optimizer_step", self._optimizer_step))
        self.bind_algorithm(algorithm)

    def post_optimizer_step(self, model: nn.Module, metrics: Optional[Dict[str, Any]] = None) -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if self.eval_ema is not None:
            self.eval_ema.step(trainable)
        if self.reference_param_ema is not None:
            self.reference_param_ema.step(trainable)
        if self.reference_adapter_ema is not None:
            if self.reference_update_timing == "optimizer_step":
                ema_success = self.reference_adapter_ema.update(model, step=self._optimizer_step)
                if metrics is not None:
                    metrics["ema_updated"] = ema_success
            elif metrics is not None:
                metrics["ema_updated"] = False
        self._optimizer_step += 1

    def post_rollout_end(self, model: nn.Module, metrics: Optional[Dict[str, Any]] = None) -> None:
        if self.reference_adapter_ema is None:
            return
        if self.reference_update_timing != "rollout_end":
            return
        ema_success = self.reference_adapter_ema.update(model, step=self._optimizer_step)
        if metrics is not None:
            metrics["ema_updated"] = ema_success

    def apply_eval_ema(self, model: nn.Module) -> bool:
        if self.eval_ema is None:
            return False
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.eval_ema.copy_ema_to(trainable, store_temp=True, grad=False)
        return True

    def restore_from_eval(self, model: nn.Module) -> bool:
        if self.eval_ema is None or self.eval_ema.temp_stored_parameters is None:
            return False
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.eval_ema.copy_temp_to(trainable)
        return True
