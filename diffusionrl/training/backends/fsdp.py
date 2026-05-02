"""FSDP2 training backend conforming to the new TrainBackend protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
from torch import nn

from diffusionrl.config.registration import register_config
from diffusionrl.config.validation import validate_precision_type
from diffusionrl.models.base import ModelBundle
from diffusionrl.training.backends.base import (
    LrSchedulerConfig,
    OptimizerConfig,
    TrainBackendConfig,
    TrainTopology,
)
from diffusionrl.training.backends.protocols import (
    LRSchedulerProtocol,
    OptimizerProtocol,
)
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


@register_config(
    group="training/backend",
    name="fsdp",
    target="diffusionrl.training.backends.fsdp.FSDPBackend",
)
@dataclass
class FSDPBackendConfig(TrainBackendConfig):
    """Config for the FSDP2 training backend.

    Pure FSDP-specific settings. ``name`` is a ClassVar (not a dataclass
    field) so validation / schema code that expects ``config.name`` keeps
    working, without polluting the frozen-dataclass init signature.
    """

    name: ClassVar[str] = "fsdp"

    cpu_offload: bool = False
    param_dtype: str = "bf16"
    mixed_precision: bool = True
    fsdp_mode: str = "full"
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        # Normalize aliases (bfloat16/bf16/fp16/...) to the canonical short form
        # so the resolved cfg stays YAML-serializable (Hydra dumps it on every
        # run) and the registry renders identically across recipes.
        self.param_dtype = validate_precision_type(self.param_dtype, field="training.backend.param_dtype")


class FSDPBackend:
    def __init__(
        self,
        config: FSDPBackendConfig,
        model_bundle: ModelBundle,
        *,
        topology: Optional[TrainTopology] = None,
    ) -> None:
        # topology is accepted for call-site symmetry with VeOmniBackend but
        # unused here — FSDP2 derives world/DP sizes from
        # ``torch.distributed.get_world_size()`` at wrap time.
        del topology
        self.config = config
        self.model_bundle: ModelBundle = model_bundle
        self.model: nn.Module = model_bundle.transformer

        self.last_grad_norm: Optional[torch.Tensor] = None

        # Capture the target device before any offload happens so onload
        # can restore params to the same place.
        self._device = self._infer_device(self.model)
        self._is_offloaded = False
        self._device_mesh: Optional[Any] = None
        self._wrap_model()

    # ------------------------------------------------------------------
    # FSDP2 wrap
    # ------------------------------------------------------------------

    def _wrap_model(self) -> None:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )

        fsdp_kwargs: Dict[str, Any] = {
            "reshard_after_forward": bool(self.config.reshard_after_forward),
        }

        if self.config.mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=parse_torch_dtype(self.config.param_dtype, field_name="training.backend.param_dtype"),
                reduce_dtype=torch.float32,
            )

        if self.config.cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

        mesh = self._create_device_mesh()
        if mesh is not None:
            fsdp_kwargs["mesh"] = mesh
            self._device_mesh = mesh

        target_modules = self._iter_target_modules()
        for module in target_modules:
            fully_shard(module, **fsdp_kwargs)

        fully_shard(self.model, **fsdp_kwargs)
        logger.info(
            "FSDPBackend: model wrapped with %s fully_shard "
            "(target_modules=%d, cpu_offload=%s, mixed_precision=%s, "
            "reshard_after_forward=%s)",
            "HSDP" if mesh is not None else "FSDP2",
            len(target_modules),
            self.config.cpu_offload,
            self.config.mixed_precision,
            fsdp_kwargs["reshard_after_forward"],
        )

    def _create_device_mesh(self) -> Optional[Any]:
        """Create a 2D DeviceMesh for HSDP (hybrid mode).

        Shards within groups of 8 GPUs (one node), replicates across nodes.
        Returns ``None`` for ``full`` mode, for world_size <= 8 (no benefit),
        or when world_size is not a multiple of 8 (no safe grouping).
        """
        fsdp_mode = str(self.config.fsdp_mode).strip().lower()
        if fsdp_mode != "hybrid":
            return None

        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None

        world_size = dist.get_world_size()
        shard_size = 8
        if world_size <= shard_size:
            logger.info(
                "FSDPBackend: hybrid requested but world_size=%d <= %d; falling back to pure FSDP.",
                world_size,
                shard_size,
            )
            return None
        if world_size % shard_size != 0:
            logger.warning(
                "FSDPBackend: hybrid requested but world_size=%d is not a multiple of %d; falling back to pure FSDP.",
                world_size,
                shard_size,
            )
            return None

        from torch.distributed.device_mesh import init_device_mesh

        replicate_size = world_size // shard_size
        mesh = init_device_mesh(
            "cuda",
            (replicate_size, shard_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        logger.info(
            "FSDPBackend: HSDP mesh dp_replicate=%d × dp_shard=%d",
            replicate_size,
            shard_size,
        )
        return mesh

    def _iter_target_modules(self) -> Tuple[nn.Module, ...]:
        if self.model_bundle is None:
            return tuple()
        if not hasattr(self.model_bundle, "get_no_split_modules"):
            return tuple()

        no_split_modules = self.model_bundle.get_no_split_modules()
        if not isinstance(no_split_modules, tuple) or not no_split_modules:
            return tuple()

        targets: list[nn.Module] = []
        for _name, module in self.model.named_modules():
            if isinstance(module, no_split_modules):
                targets.append(module)
        return tuple(targets)

    # ------------------------------------------------------------------
    # Protocol: state dict
    # ------------------------------------------------------------------

    def get_state_dict(self, *, lora_only: bool = False) -> Dict[str, Any]:
        """Return a model state dict on rank 0 (empty dict elsewhere).

        When ``lora_only`` is set, only LoRA adapter parameters are returned.
        The peft-aware path is preferred; if peft is unavailable or the model
        has no peft config, fall back to substring-matching ``"lora"`` in
        parameter keys. If neither yields any LoRA parameters, raise.
        """
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        options = self._build_state_dict_options(
            full_state_dict=True,
            cpu_offload=True,
        )
        try:
            full = dict(get_model_state_dict(self.model, options=options))
        except TypeError:
            full = dict(get_model_state_dict(self.model))

        if self._current_rank() != 0:
            return {}

        full = self._to_cpu_state_dict(full)

        if lora_only:
            peft_state = self._extract_peft_lora_state(self.model)
            if peft_state:
                return self._to_cpu_state_dict(peft_state)
            filtered = self._filter_lora_state(full)
            if filtered:
                return filtered
            raise ValueError("LoRA-only state dict requested but no LoRA parameters were found in the model.")

        return full

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a full state dict, broadcasting from rank 0 across ranks."""
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        options = self._build_state_dict_options(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=False,
        )
        try:
            set_model_state_dict(self.model, state_dict, options=options)
        except TypeError:
            set_model_state_dict(self.model, state_dict)

    @staticmethod
    def _build_state_dict_options(**kwargs: Any) -> Any:
        """Build ``StateDictOptions`` tolerating cross-version kwarg drift."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        candidates = [
            dict(kwargs),
            {k: v for k, v in kwargs.items() if k != "broadcast_from_rank0"},
            {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
            {},
        ]
        for candidate in candidates:
            try:
                return StateDictOptions(**candidate)
            except TypeError:
                continue
        return StateDictOptions()

    # ------------------------------------------------------------------
    # Protocol: clip_grad_norm
    # ------------------------------------------------------------------

    def clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip gradient norm and return the pre-clip norm.

        The protocol declares ``-> None`` but Python's structural typing
        accepts a wider return; the legacy caller logs the norm so this
        preserves that capability. The norm is also stored on
        ``self.last_grad_norm``.
        """
        grad_norm = self._do_clip_grad_norm(max_grad_norm)
        self.last_grad_norm = grad_norm
        return grad_norm

    def _do_clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        if self.config.cpu_offload:
            # Use the explicit global-norm clipping path for FSDP2 + CPU
            # offload. This avoids DTensor CPU collective limitations in
            # ``clip_grad_norm_``.
            return self._global_clip_for_sharded_grads(max_grad_norm)

        try:
            clip_fn = getattr(self.model, "clip_grad_norm_", None)
            if callable(clip_fn):
                grad_norm = clip_fn(max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm,
                )
            return self._maybe_dtensor_to_tensor(grad_norm)
        except RuntimeError as exc:
            # FSDP2 + CPU offload can surface DTensor CPU collective errors here.
            if "No backend type associated with device type cpu" not in str(exc):
                raise
            logger.warning(
                "FSDPBackend: grad clipping hit CPU DTensor backend error; "
                "falling back to explicit global-norm clipping path."
            )
            return self._global_clip_for_sharded_grads(max_grad_norm)

    def _global_clip_for_sharded_grads(self, max_grad_norm: float) -> torch.Tensor:
        import torch.distributed as dist

        grads: list[torch.Tensor] = []
        local_sq_sum = 0.0
        for param in self.model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue

            local_grad = grad
            if hasattr(local_grad, "to_local") and callable(getattr(local_grad, "to_local")):
                try:
                    local_grad = local_grad.to_local()
                except Exception:
                    pass

            if not isinstance(local_grad, torch.Tensor):
                continue

            local_sq_sum += float(torch.sum(local_grad.detach().float() ** 2).item())
            grads.append(grad)

        if not grads:
            return torch.tensor(0.0)

        reduce_device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                reduce_device = torch.device("cuda")

        total_sq = torch.tensor(local_sq_sum, device=reduce_device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)

        global_norm = float(torch.sqrt(total_sq).item())
        clip_coef = float(max_grad_norm) / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)

        return torch.tensor(global_norm, device=reduce_device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Protocol: offload / onload (model state only)
    # ------------------------------------------------------------------

    def offload(self) -> None:
        """Move model params and grads to CPU. Idempotent."""
        if self._is_offloaded:
            return

        cpu = torch.device("cpu")
        for param in self.model.parameters():
            local = self._local_or_self(param.data)
            param.data = local.to(cpu, non_blocking=False)
            if param.grad is not None:
                local_grad = self._local_or_self(param.grad)
                param.grad = local_grad.to(cpu, non_blocking=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_offloaded = True
        logger.debug("FSDPBackend: offloaded params/grads to CPU")

    def onload(self) -> None:
        """Move model params and grads back to the recorded GPU device. Idempotent."""
        if not self._is_offloaded:
            return

        target = self._device
        for param in self.model.parameters():
            local = self._local_or_self(param.data)
            param.data = local.to(target, non_blocking=False)
            if param.grad is not None:
                local_grad = self._local_or_self(param.grad)
                param.grad = local_grad.to(target, non_blocking=False)

        self._is_offloaded = False
        logger.debug("FSDPBackend: onloaded params/grads to %s", target)

    # ------------------------------------------------------------------
    # Protocol: build_optimizer / build_scheduler
    # ------------------------------------------------------------------

    def build_optimizer(self, config: OptimizerConfig) -> Optional[OptimizerProtocol]:
        """Default to the factory-level optimizer (torch AdamW)."""
        del config
        return None

    def build_scheduler(
        self,
        config: LrSchedulerConfig,
        optimizer: OptimizerProtocol,
    ) -> Optional[LRSchedulerProtocol]:
        """Default to the factory-level LR scheduler."""
        del config, optimizer
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_dtensor_to_tensor(value: Any) -> Any:
        if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
            try:
                return value.full_tensor()
            except Exception:
                return value
        return value

    @staticmethod
    def _to_cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for key, value in state_dict.items():
            tensor_or_obj = FSDPBackend._maybe_dtensor_to_tensor(value)
            if isinstance(tensor_or_obj, torch.Tensor):
                converted[key] = tensor_or_obj.detach().cpu()
            else:
                converted[key] = tensor_or_obj
        return converted

    @staticmethod
    def _filter_lora_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return only entries whose key contains ``"lora"`` (case-insensitive)."""
        return {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}

    @staticmethod
    def _extract_peft_lora_state(model: nn.Module) -> Dict[str, Any]:
        """Return the peft LoRA adapter state dict, or ``{}`` if peft is unavailable.

        Unwraps ``model.module`` once (covers any single-layer wrapper like
        ``PeftModel.base_model``-style wrappers), iterates every adapter
        registered in ``peft_config``, and falls through to the active
        adapter (default ``"default"``) when no adapters are registered.
        """
        try:
            from peft.utils import get_peft_model_state_dict
        except Exception:
            return {}

        base_model = model.module if hasattr(model, "module") else model
        adapter_names: list[str] = []
        if hasattr(base_model, "peft_config"):
            adapter_names = list(base_model.peft_config.keys())
        if not adapter_names:
            adapter_names = [getattr(base_model, "active_adapter", "default")]

        lora_state: Dict[str, Any] = {}
        for adapter_name in adapter_names:
            lora_state.update(get_peft_model_state_dict(base_model, adapter_name=adapter_name))
        return lora_state

    @staticmethod
    def _local_or_self(tensor: torch.Tensor) -> torch.Tensor:
        """Return the local shard for DTensors, else the tensor itself."""
        if hasattr(tensor, "to_local") and callable(getattr(tensor, "to_local")):
            try:
                return tensor.to_local()
            except Exception:
                return tensor
        return tensor

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        for param in model.parameters():
            return param.device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    @staticmethod
    def _current_rank() -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0


__all__ = [
    "FSDPBackendConfig",
    "FSDPBackend",
]
