"""VeOmni training backend conforming to the new TrainBackend protocol.

Ports ``backends/veomni_native.py`` to the protocol-based system:
- No actor argument on any method — the backend owns its model/bundle.
- Parallel-state init and model wrap happen in ``__init__`` instead of in
  ``before_model_load`` / ``wrap_model`` lifecycle hooks.
- ``build_optimizer`` / ``build_scheduler`` are invoked by
  ``diffusionrl.training.factories`` and return duck-typed
  ``OptimizerProtocol`` / ``LRSchedulerProtocol`` instances (VeOmni may
  return ``MultiOptimizer`` / ``MultiLRScheduler`` under ExtraParallel+FSDP2).
- ``offload`` / ``onload`` are no-ops — VeOmni manages its own memory via
  its FSDP2 offload policy.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from torch import nn

from diffusionrl.config.registration import register_config
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

logger = logging.getLogger(__name__)


@register_config(
    group="training/backend",
    name="veomni",
    target="diffusionrl.training.backends.veomni.VeOmniBackend",
)
@dataclass
class VeOmniBackendConfig(TrainBackendConfig):
    """Config for the VeOmni training backend.

    Pure VeOmni settings. ``name`` is a ClassVar (not a dataclass field) so
    validation / schema code that expects ``config.name`` keeps working,
    without polluting the frozen-dataclass init signature. Fields mirror
    ``VeOmniTrainBackendConfig`` from the retired old backend minus the
    launch-spec hints (``actor_class_path``, ``num_gpus_per_actor``,
    ``runtime_env``, ``actor_kwargs``) which were only consumed by
    ``TrainBackendLaunchSpec`` and have no equivalent in the new system.
    """

    name: ClassVar[str] = "veomni"

    # Topology / parallel mode
    # All parallel dims (dp_size, dp_replicate_size, dp_shard_size, tp_size,
    # pp_size, sp_size, ep_size, cp_size) come from the TrainTopology
    # runtime dep injected by the caller at build() time. Only
    # VeOmni-specific knobs live here.
    data_parallel_mode: str = "fsdp2"

    # VeOmni parallelize knobs
    enable_full_shard: bool = True
    enable_reshard_after_forward: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_reentrant: bool = False
    enable_forward_prefetch: bool = False
    enable_fsdp_offload: bool = False
    init_device: str = "meta"
    broadcast_model_weights_from_rank0: bool = True
    basic_modules: Optional[List[str]] = None
    parallelize_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Weights path control (important for non-HF diffusion model layouts)
    weights_path: Optional[str] = None
    weights_path_mode: str = "transformer_subdir"

    # Optimizer / scheduler knobs
    optimizer_type: str = "adamw"
    fused_optimizer: bool = False
    no_decay_modules: Optional[List[str]] = None
    no_decay_params: Optional[List[str]] = None
    lr_decay_ratio: float = 1.0
    lr_min: float = 1e-7
    lr_start: float = 0.0

    # Import path control for local VeOmni repo
    veomni_repo_path: Optional[str] = None


class VeOmniBackend:
    def __init__(
        self,
        config: VeOmniBackendConfig,
        model_bundle: ModelBundle,
        *,
        topology: TrainTopology,
    ) -> None:
        dp_mode = str(config.data_parallel_mode or "fsdp2").strip().lower()
        if dp_mode != "fsdp2":
            raise ValueError(f"VeOmniBackend currently supports data_parallel_mode='fsdp2' only. Got: {dp_mode!r}")

        self.config = config
        self.model_bundle: ModelBundle = model_bundle
        self.model: nn.Module = model_bundle.transformer
        self._topology = topology
        self._dp_mode = dp_mode
        self._last_optimizer_lr: Optional[float] = None
        self._runtime_init_device: Optional[str] = None

        self._init_parallel_state()
        self._wrap_model()

    # ------------------------------------------------------------------
    # VeOmni import plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_veomni_repo_from_workspace() -> Optional[str]:
        # .../<workspace>/diffusionrl/diffusionrl/training/backends/veomni.py
        # sibling candidate is .../<workspace>/VeOmni
        this_file = Path(__file__).resolve()
        diffusionrl_root = this_file.parents[3]
        sibling = diffusionrl_root.parent / "VeOmni"
        if sibling.exists() and (sibling / "veomni").exists():
            return str(sibling)
        return None

    def _ensure_veomni_import_path(self) -> None:
        candidate_paths: List[str] = []
        repo_path = self.config.veomni_repo_path
        if isinstance(repo_path, str) and repo_path.strip():
            candidate_paths.append(repo_path.strip())
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
                "VeOmni backend requires importable `veomni` package. "
                "Provide backend kwarg `veomni_repo_path` or set PYTHONPATH/VEOMNI_REPO_PATH."
            ) from exc

        return (
            init_parallel_state,
            build_parallelize_model,
            build_optimizer,
            build_lr_scheduler,
            veomni_clip_grad_norm,
        )

    # ------------------------------------------------------------------
    # Topology + initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _current_world_size() -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1

    @staticmethod
    def _current_rank() -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    def _resolve_dp_sizes(self) -> Tuple[int, int, int]:
        topo = self._topology
        world_size = self._current_world_size()
        dp_size = int(topo.dp_size) if topo.dp_size is not None else int(world_size)
        dp_replicate = int(topo.dp_replicate_size)
        dp_shard = (
            int(topo.dp_shard_size) if topo.dp_shard_size is not None else max(1, dp_size // max(1, dp_replicate))
        )

        if dp_replicate * dp_shard != dp_size:
            raise ValueError(
                "Invalid VeOmni parallel config: dp_replicate_size * dp_shard_size "
                f"must equal dp_size. Got {dp_replicate} * {dp_shard} != {dp_size}."
            )
        return dp_size, dp_replicate, dp_shard

    def _init_parallel_state(self) -> None:
        init_parallel_state, *_ = self._veomni_apis()
        dp_size, dp_replicate, dp_shard = self._resolve_dp_sizes()
        topo = self._topology
        runtime_init_device = str(self.config.init_device)
        if dp_size <= 1 and runtime_init_device in {"meta", "cpu"}:
            logger.warning(
                "VeOmni backend detected dp_size=%s. Falling back init_device "
                "from %s to 'cuda' for single-rank execution.",
                dp_size,
                runtime_init_device,
            )
            runtime_init_device = "cuda"
        self._runtime_init_device = runtime_init_device
        init_parallel_state(
            dp_size=int(dp_size),
            dp_replicate_size=int(dp_replicate),
            dp_shard_size=int(dp_shard),
            tp_size=int(topo.tp_size),
            ep_size=int(topo.ep_size),
            pp_size=int(topo.pp_size),
            cp_size=int(topo.cp_size),
            ulysses_size=int(topo.sp_size),
            dp_mode=self._dp_mode,
        )

    def _resolve_basic_modules(self) -> List[str]:
        if isinstance(self.config.basic_modules, (list, tuple)) and self.config.basic_modules:
            return [str(name) for name in self.config.basic_modules]

        if not hasattr(self.model_bundle, "get_no_split_modules"):
            return []

        no_split = self.model_bundle.get_no_split_modules()
        if not isinstance(no_split, tuple):
            return []
        result: List[str] = []
        for cls in no_split:
            if hasattr(cls, "__name__"):
                result.append(str(cls.__name__))
        return result

    def _resolve_weights_path(self) -> str:
        override = self.config.weights_path
        if isinstance(override, str) and override.strip():
            return os.path.abspath(os.path.expanduser(override.strip()))

        pretrained_path = getattr(self.model_bundle, "pretrained_path", None)
        if not pretrained_path:
            raise ValueError(
                "VeOmni backend requires checkpoint path for weight reload. Provide backend kwarg `weights_path`."
            )

        base_path = os.path.abspath(os.path.expanduser(str(pretrained_path)))
        if self.config.weights_path_mode == "transformer_subdir":
            candidate = os.path.join(base_path, "transformer")
            if os.path.isdir(candidate):
                return candidate
        return base_path

    def _wrap_model(self) -> None:
        _init_parallel_state, build_parallelize_model, *_ = self._veomni_apis()
        basic_modules = self._resolve_basic_modules()
        weights_path = self._resolve_weights_path()
        init_device = str(self._runtime_init_device or self.config.init_device)

        parallelize_extra_kwargs = dict(self.config.parallelize_kwargs or {})

        logger.info(
            "Rank %s: VeOmni wrap start (dp_mode=%s, init_device=%s, weights_path=%s, basic_modules=%s)",
            self._current_rank(),
            self._dp_mode,
            init_device,
            weights_path,
            basic_modules,
        )

        model = build_parallelize_model(
            model=self.model,
            weights_path=weights_path,
            enable_full_shard=bool(self.config.enable_full_shard),
            enable_reshard_after_forward=bool(self.config.enable_reshard_after_forward),
            enable_mixed_precision=bool(self.config.enable_mixed_precision),
            enable_gradient_checkpointing=bool(self.config.enable_gradient_checkpointing),
            basic_modules=basic_modules,
            init_device=init_device,
            broadcast_model_weights_from_rank0=bool(self.config.broadcast_model_weights_from_rank0),
            enable_reentrant=bool(self.config.enable_reentrant),
            enable_forward_prefetch=bool(self.config.enable_forward_prefetch),
            enable_fsdp_offload=bool(self.config.enable_fsdp_offload),
            **parallelize_extra_kwargs,
        )
        self.model = model

    # ------------------------------------------------------------------
    # Protocol: state dict
    # ------------------------------------------------------------------

    @staticmethod
    def _state_dict_apis() -> Tuple[Any, Any, Any]:
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                set_model_state_dict,
            )
        except Exception as exc:
            raise RuntimeError("VeOmni backend requires torch distributed checkpoint state-dict APIs.") from exc
        return StateDictOptions, get_model_state_dict, set_model_state_dict

    def _build_state_dict_options(self, **kwargs: Any) -> Any:
        StateDictOptions, _a, _b = self._state_dict_apis()
        # Older torch releases reject newer StateDictOptions kwargs; fall back
        # progressively so the backend keeps working on pinned torch versions.
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

    def get_state_dict(self, *, lora_only: bool = False) -> Dict[str, Any]:
        _options, get_model_state_dict, _set = self._state_dict_apis()
        options = self._build_state_dict_options(
            full_state_dict=True,
            cpu_offload=True,
        )
        try:
            state_dict = dict(get_model_state_dict(self.model, options=options))
        except TypeError:
            state_dict = dict(get_model_state_dict(self.model))

        if self._current_rank() != 0:
            return {}
        state_dict = self._to_cpu_state_dict(state_dict)

        if lora_only:
            filtered = {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}
            if not filtered:
                raise ValueError(
                    "LoRA-only state dict requested but no LoRA parameters were found in the VeOmni state dict."
                )
            return filtered
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        _options, _get, set_model_state_dict = self._state_dict_apis()
        options = self._build_state_dict_options(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=False,
        )
        try:
            set_model_state_dict(self.model, state_dict, options=options)
        except TypeError:
            set_model_state_dict(self.model, state_dict)

    # ------------------------------------------------------------------
    # Protocol: clip_grad_norm
    # ------------------------------------------------------------------

    def clip_grad_norm(self, max_grad_norm: float) -> None:
        *_a, veomni_clip_grad_norm = self._veomni_apis()
        veomni_clip_grad_norm(self.model, max_norm=float(max_grad_norm))

    # ------------------------------------------------------------------
    # Protocol: offload / onload
    # ------------------------------------------------------------------

    def offload(self) -> None:
        # VeOmni manages its own memory via FSDP2 offload policy.
        return

    def onload(self) -> None:
        return

    # ------------------------------------------------------------------
    # Protocol: build_optimizer / build_scheduler
    # ------------------------------------------------------------------

    def build_optimizer(self, config: OptimizerConfig) -> Optional[OptimizerProtocol]:
        *_a, _b, build_optimizer, _c, _d = self._veomni_apis()

        lr = float(config.learning_rate)
        betas = (float(config.adam_beta1), float(config.adam_beta2))
        eps = float(config.adam_epsilon)
        weight_decay = float(config.weight_decay)

        self._last_optimizer_lr = lr
        return build_optimizer(
            model=self.model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=bool(self.config.fused_optimizer),
            optimizer_type=self.config.optimizer_type,
            no_decay_modules=self.config.no_decay_modules,
            no_decay_params=self.config.no_decay_params,
        )

    def build_scheduler(
        self,
        config: LrSchedulerConfig,
        optimizer: OptimizerProtocol,
    ) -> Optional[LRSchedulerProtocol]:
        *_a, _b, _c, build_lr_scheduler, _d = self._veomni_apis()

        total_steps = max(1, int(config.total_steps))
        warmup_steps = max(0, int(config.warmup_steps))
        warmup_ratio = min(1.0, float(warmup_steps) / float(total_steps))

        if self._last_optimizer_lr is None:
            raise RuntimeError(
                "VeOmni scheduler requires optimizer lr to be resolved before "
                "scheduler creation (call build_optimizer first)."
            )
        lr = float(self._last_optimizer_lr)
        return build_lr_scheduler(
            optimizer=optimizer,
            train_steps=total_steps,
            lr=lr,
            lr_decay_style=str(config.type),
            lr_decay_ratio=float(self.config.lr_decay_ratio),
            lr_warmup_ratio=float(warmup_ratio),
            lr_min=float(self.config.lr_min),
            lr_start=float(self.config.lr_start),
        )

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
            tensor_or_obj = VeOmniBackend._maybe_dtensor_to_tensor(value)
            if isinstance(tensor_or_obj, torch.Tensor):
                converted[key] = tensor_or_obj.detach().cpu()
            else:
                converted[key] = tensor_or_obj
        return converted


__all__ = [
    "VeOmniBackendConfig",
    "VeOmniBackend",
]
