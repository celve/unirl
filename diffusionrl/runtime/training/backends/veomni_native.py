"""Native VeOmni training backend (opt-in via train_backend_path).

This backend intentionally does not replace the built-in ``veomni`` backend.
Use it explicitly with:

    --train-backend veomni \
    --train-backend-path diffusionrl.runtime.training.backends.veomni_native.VeOmniNativeTrainBackend

Design goals:
- Keep default diffusionRL mainline behavior unchanged.
- Provide a true VeOmni API integration path (parallelize/optimizer/lr-scheduler/grad clip).
- Fail fast when VeOmni runtime is unavailable or incompatible.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .base import TrainBackend, TrainBackendCapabilities, TrainTopology

logger = logging.getLogger(__name__)


def _as_optional_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if value < 1:
        return None
    return value


class VeOmniNativeTrainBackend(TrainBackend):
    """VeOmni-native backend (FSDP2-focused, EP/SP capable when model supports it)."""

    BACKEND_NAME = "veomni_native"

    def __init__(self, *, backend_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(backend_kwargs=backend_kwargs)
        kwargs = dict(self.backend_kwargs)

        self._dp_mode = str(kwargs.pop("data_parallel_mode", "fsdp2") or "fsdp2").strip().lower()
        if self._dp_mode != "fsdp2":
            raise ValueError(
                "VeOmniNativeTrainBackend currently supports data_parallel_mode='fsdp2' only. "
                f"Got: {self._dp_mode!r}"
            )

        # Parallel topology knobs.
        self._dp_size_hint = _as_optional_int(kwargs.pop("dp_size", None))
        self._dp_replicate_size_hint = _as_optional_int(kwargs.pop("dp_replicate_size", None)) or 1
        self._dp_shard_size_hint = _as_optional_int(kwargs.pop("dp_shard_size", None))
        self._tp_size = _as_optional_int(kwargs.pop("tp_size", None)) or 1
        self._pp_size = _as_optional_int(kwargs.pop("pp_size", None)) or 1
        self._sp_size = _as_optional_int(kwargs.pop("sp_size", None)) or 1
        self._ep_size = _as_optional_int(kwargs.pop("ep_size", None)) or 1
        self._cp_size = _as_optional_int(kwargs.pop("cp_size", None)) or 1

        # VeOmni parallelize knobs.
        self._enable_full_shard = bool(kwargs.pop("enable_full_shard", True))
        self._enable_reshard_after_forward = bool(kwargs.pop("enable_reshard_after_forward", True))
        self._enable_mixed_precision = bool(kwargs.pop("enable_mixed_precision", True))
        self._enable_gradient_checkpointing = bool(kwargs.pop("enable_gradient_checkpointing", False))
        self._enable_reentrant = bool(kwargs.pop("enable_reentrant", False))
        self._enable_forward_prefetch = bool(kwargs.pop("enable_forward_prefetch", False))
        self._enable_fsdp_offload = bool(kwargs.pop("enable_fsdp_offload", False))
        self._init_device = str(kwargs.pop("init_device", "meta") or "meta").strip().lower()
        self._broadcast_from_rank0 = bool(kwargs.pop("broadcast_model_weights_from_rank0", True))
        self._basic_modules = kwargs.pop("basic_modules", None)

        # Weight path control (important for non-HF diffusion model layouts).
        self._weights_path_override = kwargs.pop("weights_path", None)
        self._weights_path_mode = str(
            kwargs.pop("weights_path_mode", "transformer_subdir") or "transformer_subdir"
        ).strip().lower()

        # Optimizer/lr scheduler knobs.
        self._optimizer_type = str(kwargs.pop("optimizer_type", "adamw") or "adamw")
        self._optimizer_fused = bool(kwargs.pop("fused_optimizer", False))
        self._no_decay_modules = kwargs.pop("no_decay_modules", None)
        self._no_decay_params = kwargs.pop("no_decay_params", None)
        self._lr_decay_ratio = float(kwargs.pop("lr_decay_ratio", 1.0))
        self._lr_min = float(kwargs.pop("lr_min", 1e-7))
        self._lr_start = float(kwargs.pop("lr_start", 0.0))

        # Import path control for local VeOmni repo.
        self._veomni_repo_path = kwargs.pop("veomni_repo_path", None)

        # Advanced pass-through for build_parallelize_model only.
        raw_parallelize_kwargs = kwargs.pop("parallelize_kwargs", {})
        if raw_parallelize_kwargs is None:
            raw_parallelize_kwargs = {}
        if not isinstance(raw_parallelize_kwargs, dict):
            raise TypeError("parallelize_kwargs must be a dict when provided.")
        self._parallelize_extra_kwargs = dict(raw_parallelize_kwargs)

        # Keep unknown options explicit for easier debugging.
        self._unused_kwargs = dict(kwargs)
        if self._unused_kwargs:
            logger.warning(
                "VeOmniNativeTrainBackend received unused backend kwargs: %s",
                sorted(self._unused_kwargs.keys()),
            )

        self._last_optimizer_lr: Optional[float] = None
        self._runtime_init_device: Optional[str] = None

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        return TrainBackendCapabilities(
            name=cls.BACKEND_NAME,
            distributed_backend="nccl",
            supports_training_actor_sampling=True,
            buffer_partition_mode="data_parallel",
            supports_state_dict_export=True,
            supports_custom_optimizer=True,
            supports_custom_scheduler=True,
            supports_custom_train_step=False,
            supports_backend_managed_offload=False,
            preferred_weight_transport="checkpoint_path",
            preferred_weight_export_format="state_dict",
            supported_weight_export_formats=("state_dict",),
            notes=(
                "Experimental native VeOmni integration path. "
                "Use only via explicit train_backend_path to avoid affecting default mainline."
            ),
        )

    @staticmethod
    def _candidate_veomni_repo_from_workspace() -> Optional[str]:
        # .../diffusionRL/diffusionrl/runtime/training/backends/veomni_native.py
        # sibling candidate is .../VeOmni
        this_file = Path(__file__).resolve()
        diffusionrl_root = this_file.parents[5]
        sibling = diffusionrl_root.parent / "VeOmni"
        if sibling.exists() and (sibling / "veomni").exists():
            return str(sibling)
        return None

    def _ensure_veomni_import_path(self) -> None:
        candidate_paths = []
        if isinstance(self._veomni_repo_path, str) and self._veomni_repo_path.strip():
            candidate_paths.append(self._veomni_repo_path.strip())
        env_path = os.getenv("VEOMNI_REPO_PATH", "").strip()
        if env_path:
            candidate_paths.append(env_path)
        auto_path = self._candidate_veomni_repo_from_workspace()
        if auto_path:
            candidate_paths.append(auto_path)

        for path in candidate_paths:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if not os.path.isdir(abs_path):
                continue
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
                logger.info("VeOmni import path injected: %s", abs_path)
            break

    def _veomni_apis(self) -> Tuple[Any, Any, Any, Any, Any]:
        self._ensure_veomni_import_path()
        # Avoid pulling VeOmni custom kernel patch path unless caller explicitly sets otherwise.
        os.environ.setdefault("MODELING_BACKEND", "hf")

        try:
            from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
            from veomni.distributed.parallel_state import init_parallel_state
            from veomni.distributed.torch_parallelize import build_parallelize_model
            from veomni.optim.lr_scheduler import build_lr_scheduler
            from veomni.optim.optimizer import build_optimizer
        except Exception as exc:
            raise RuntimeError(
                "VeOmni native backend requires importable `veomni` package. "
                "Provide backend kwarg `veomni_repo_path` or set PYTHONPATH/VEOMNI_REPO_PATH."
            ) from exc

        return (
            init_parallel_state,
            build_parallelize_model,
            build_optimizer,
            build_lr_scheduler,
            veomni_clip_grad_norm,
        )

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
            tensor_or_obj = VeOmniNativeTrainBackend._maybe_dtensor_to_tensor(value)
            if isinstance(tensor_or_obj, torch.Tensor):
                converted[key] = tensor_or_obj.detach().cpu()
            else:
                converted[key] = tensor_or_obj
        return converted

    @staticmethod
    def _filter_lora_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}

    def _resolve_dp_sizes(self, actor: Any) -> Tuple[int, int, int]:
        world_size = int(getattr(actor, "world_size", 1))
        dp_size = int(self._dp_size_hint or world_size)
        dp_replicate = int(self._dp_replicate_size_hint or 1)
        dp_shard = int(self._dp_shard_size_hint or max(1, dp_size // max(1, dp_replicate)))

        if dp_replicate * dp_shard != dp_size:
            raise ValueError(
                "Invalid VeOmni parallel config: dp_replicate_size * dp_shard_size must equal dp_size. "
                f"Got {dp_replicate} * {dp_shard} != {dp_size}."
            )
        return dp_size, dp_replicate, dp_shard

    def _resolve_basic_modules(self, actor: Any) -> list[str]:
        if isinstance(self._basic_modules, (list, tuple)) and self._basic_modules:
            return [str(name) for name in self._basic_modules]

        model_bundle = getattr(actor, "model_bundle", None)
        if model_bundle is None or not hasattr(model_bundle, "get_no_split_modules"):
            return []

        no_split = model_bundle.get_no_split_modules()
        if not isinstance(no_split, tuple):
            return []
        result: list[str] = []
        for cls in no_split:
            if hasattr(cls, "__name__"):
                result.append(str(cls.__name__))
        return result

    def _resolve_weights_path(self, actor: Any) -> str:
        if isinstance(self._weights_path_override, str) and self._weights_path_override.strip():
            return os.path.abspath(os.path.expanduser(self._weights_path_override.strip()))

        pretrained_path = None
        model_bundle = getattr(actor, "model_bundle", None)
        if model_bundle is not None:
            pretrained_path = getattr(model_bundle, "pretrained_path", None)
        if not pretrained_path:
            raise ValueError(
                "VeOmni native backend requires checkpoint path for weight reload. "
                "Provide backend kwarg `weights_path`."
            )

        base_path = os.path.abspath(os.path.expanduser(str(pretrained_path)))
        if self._weights_path_mode == "transformer_subdir":
            candidate = os.path.join(base_path, "transformer")
            if os.path.isdir(candidate):
                return candidate
        return base_path

    def uses_sharded_model(self) -> bool:
        return True

    def data_parallel_size(self, actor: Any) -> int:
        dp_size, _dp_replicate, _dp_shard = self._resolve_dp_sizes(actor)
        return int(dp_size)

    def topology(self, actor: Any) -> TrainTopology:
        world_size = int(getattr(actor, "world_size", 1))
        dp_size, dp_replicate, dp_shard = self._resolve_dp_sizes(actor)
        return TrainTopology(
            world_size=world_size,
            dp_size=dp_size,
            dp_replicate_size=dp_replicate,
            dp_shard_size=dp_shard,
            tp_size=int(self._tp_size),
            pp_size=int(self._pp_size),
            sp_size=int(self._sp_size),
            ep_size=int(self._ep_size),
            data_partition_axis="dp",
        )

    def before_model_load(self, actor: Any) -> None:
        # Keep diffusionRL model loading path unchanged (no legacy fsdp cpu-offload behavior here).
        actor._fsdp_cpu_offload = False

        init_parallel_state, *_ = self._veomni_apis()
        dp_size, dp_replicate, dp_shard = self._resolve_dp_sizes(actor)
        runtime_init_device = str(self._init_device)
        if dp_size <= 1 and runtime_init_device in {"meta", "cpu"}:
            logger.warning(
                "VeOmni native backend detected dp_size=%s. "
                "Falling back init_device from %s to 'cuda' for single-rank execution.",
                dp_size,
                runtime_init_device,
            )
            runtime_init_device = "cuda"
        self._runtime_init_device = runtime_init_device
        init_parallel_state(
            dp_size=int(dp_size),
            dp_replicate_size=int(dp_replicate),
            dp_shard_size=int(dp_shard),
            tp_size=int(self._tp_size),
            ep_size=int(self._ep_size),
            pp_size=int(self._pp_size),
            cp_size=int(self._cp_size),
            ulysses_size=int(self._sp_size),
            dp_mode=self._dp_mode,
        )

    def wrap_model(self, actor: Any) -> None:
        _init_parallel_state, build_parallelize_model, *_ = self._veomni_apis()
        basic_modules = self._resolve_basic_modules(actor)
        weights_path = self._resolve_weights_path(actor)
        init_device = str(self._runtime_init_device or self._init_device)

        logger.info(
            "Rank %s: VeOmni native wrap start (dp_mode=%s, init_device=%s, weights_path=%s, basic_modules=%s)",
            actor.rank,
            self._dp_mode,
            init_device,
            weights_path,
            basic_modules,
        )

        model = build_parallelize_model(
            model=actor.model,
            weights_path=weights_path,
            enable_full_shard=self._enable_full_shard,
            enable_reshard_after_forward=self._enable_reshard_after_forward,
            enable_mixed_precision=self._enable_mixed_precision,
            enable_gradient_checkpointing=self._enable_gradient_checkpointing,
            basic_modules=basic_modules,
            init_device=init_device,
            broadcast_model_weights_from_rank0=self._broadcast_from_rank0,
            enable_reentrant=self._enable_reentrant,
            enable_forward_prefetch=self._enable_forward_prefetch,
            enable_fsdp_offload=self._enable_fsdp_offload,
            **self._parallelize_extra_kwargs,
        )
        actor.model = model

    def build_optimizer(self, actor: Any, optimizer_config: Dict[str, Any]) -> Any:
        *_a, _b, build_optimizer, _c, _d = self._veomni_apis()

        lr = float(optimizer_config.get("learning_rate", optimizer_config.get("lr", 1e-6)))
        betas = (
            float(optimizer_config.get("adam_beta1", 0.9)),
            float(optimizer_config.get("adam_beta2", 0.999)),
        )
        eps = float(optimizer_config.get("adam_epsilon", 1e-8))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))

        self._last_optimizer_lr = lr
        return build_optimizer(
            model=actor.model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=bool(self._optimizer_fused),
            optimizer_type=self._optimizer_type,
            no_decay_modules=self._no_decay_modules,
            no_decay_params=self._no_decay_params,
        )

    def build_scheduler(self, actor: Any, scheduler_config: Dict[str, Any]) -> Any:
        *_a, _b, _c, build_lr_scheduler, _d = self._veomni_apis()

        total_steps = max(1, int(scheduler_config.get("total_steps", 1000)))
        warmup_steps = max(0, int(scheduler_config.get("warmup_steps", 0)))
        warmup_ratio = min(1.0, float(warmup_steps) / float(total_steps))

        lr = float(self._last_optimizer_lr or scheduler_config.get("lr", 1e-6))
        lr_decay_style = str(scheduler_config.get("type", "constant"))
        return build_lr_scheduler(
            optimizer=actor.optimizer,
            train_steps=total_steps,
            lr=lr,
            lr_decay_style=lr_decay_style,
            lr_decay_ratio=float(self._lr_decay_ratio),
            lr_warmup_ratio=float(warmup_ratio),
            lr_min=float(self._lr_min),
            lr_start=float(self._lr_start),
        )

    def clip_grad_norm(
        self,
        actor: Any,
        *,
        model: Any,
        max_grad_norm: float,
    ) -> Any:
        *_a, veomni_clip_grad_norm = self._veomni_apis()
        del actor
        return veomni_clip_grad_norm(model, max_norm=float(max_grad_norm))

    @staticmethod
    def _state_dict_apis() -> Tuple[Any, Any, Any]:
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                set_model_state_dict,
            )
        except Exception as exc:
            raise RuntimeError(
                "VeOmni native backend requires torch distributed checkpoint state-dict APIs."
            ) from exc
        return StateDictOptions, get_model_state_dict, set_model_state_dict

    def _build_state_dict_options(self, **kwargs: Any) -> Any:
        StateDictOptions, _a, _b = self._state_dict_apis()
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

    def get_state_dict(
        self,
        actor: Any,
        *,
        lora_only: bool = False,
        rank0_only: bool = True,
    ) -> Dict[str, Any]:
        _state_dict_options, get_model_state_dict, _set_model_state_dict = self._state_dict_apis()
        options = self._build_state_dict_options(
            full_state_dict=True,
            cpu_offload=True,
        )
        try:
            state_dict = dict(get_model_state_dict(actor.model, options=options))
        except TypeError:
            state_dict = dict(get_model_state_dict(actor.model))

        if rank0_only and actor.rank != 0:
            return {}
        state_dict = self._to_cpu_state_dict(state_dict)

        if lora_only:
            lora_state = self._filter_lora_state(state_dict)
            if lora_state:
                return lora_state
            logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")
        return state_dict

    def load_state_dict(self, actor: Any, state_dict: Dict[str, Any]) -> None:
        _state_dict_options, _get_model_state_dict, set_model_state_dict = self._state_dict_apis()
        options = self._build_state_dict_options(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=False,
        )
        try:
            set_model_state_dict(actor.model, state_dict, options=options)
        except TypeError:
            set_model_state_dict(actor.model, state_dict)

    def broadcast_parameters(self, actor: Any) -> None:
        # FSDP2 shard synchronization happens during optimizer steps.
        del actor


__all__ = ["VeOmniNativeTrainBackend"]
