"""FSDP2 training backend implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from diffusionrl.utils.dtypes import parse_torch_dtype

from .base import (
    BaseTrainBackendConfig,
    TrainBackend,
    TrainBackendCapabilities,
    TrainTopology,
)
from .registry import register_train_backend

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FSDPTrainBackendConfig(BaseTrainBackendConfig):
    name: str = "fsdp"
    cpu_offload: bool = False
    param_dtype: str = "bf16"
    use_fsdp: bool = True
    mixed_precision: bool = True


@register_train_backend(component_name="fsdp", component_cfg=FSDPTrainBackendConfig)
class FSDPTrainBackend(TrainBackend):
    """FSDP2 backend (composable fully_shard).

    Notes:
    - Requires a torch build exposing `torch.distributed.fsdp.fully_shard`
      and distributed checkpoint state-dict helpers.
    - `use_fsdp=false` still keeps no-wrap debug behavior for local validation.
    """

    BACKEND_NAME = "fsdp"

    def __init__(self, config: FSDPTrainBackendConfig) -> None:
        super().__init__(config)
        self._use_fsdp = bool(config.use_fsdp)
        self._fsdp_config: Dict[str, Any] = {
            "cpu_offload": bool(config.cpu_offload),
            "param_dtype": str(config.param_dtype),
            "mixed_precision": bool(config.mixed_precision),
        }

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        return TrainBackendCapabilities(
            name=cls.BACKEND_NAME,
            distributed_backend="nccl",
            supports_training_actor_sampling=True,
            buffer_partition_mode="data_parallel",
            supports_state_dict_export=True,
            supports_custom_optimizer=False,
            supports_custom_scheduler=False,
            supports_custom_train_step=False,
            supports_backend_managed_offload=False,
            preferred_weight_export_format="state_dict",
            preferred_weight_export_format_by_rollout_engine={"sglang": "sglang_transformer_safetensors"},
            supported_weight_export_formats=("state_dict", "sglang_transformer_safetensors"),
            notes=(
                "Default backend. Uses FSDP2 fully_shard path. "
                "Requires torch with composable FSDP2 + distributed checkpoint state-dict APIs "
                "(recommended torch>=2.6)."
            ),
        )

    def uses_sharded_model(self) -> bool:
        return bool(self._use_fsdp)

    def before_model_load(self, actor: Any) -> None:
        actor._fsdp_cpu_offload = bool(self._fsdp_config.get("cpu_offload", False) and self._use_fsdp)

    @staticmethod
    def _fsdp2_runtime_apis() -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Resolve FSDP2 runtime symbols lazily to keep import-time compatibility."""
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                set_model_state_dict,
            )
            from torch.distributed.fsdp import (
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
        except Exception as exc:
            raise RuntimeError(
                "train_backend='fsdp' now targets FSDP2, but required runtime APIs are missing. "
                "Install torch>=2.4 (recommended >=2.6) with distributed checkpoint support."
            ) from exc

        return fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, get_model_state_dict, set_model_state_dict, StateDictOptions

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
            tensor_or_obj = FSDPTrainBackend._maybe_dtensor_to_tensor(value)
            if isinstance(tensor_or_obj, torch.Tensor):
                converted[key] = tensor_or_obj.detach().cpu()
            else:
                converted[key] = tensor_or_obj
        return converted

    @staticmethod
    def _filter_lora_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}

    @staticmethod
    def _extract_peft_lora_state(model: Any) -> Dict[str, Any]:
        try:
            from peft.utils import get_peft_model_state_dict
        except Exception:
            return {}

        base_model = model.module if hasattr(model, "module") else model
        adapter_names = []
        if hasattr(base_model, "peft_config"):
            adapter_names = list(base_model.peft_config.keys())
        if not adapter_names:
            adapter_names = [getattr(base_model, "active_adapter", "default")]

        lora_state: Dict[str, Any] = {}
        for adapter_name in adapter_names:
            lora_state.update(get_peft_model_state_dict(base_model, adapter_name=adapter_name))
        return lora_state

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        return model.module if hasattr(model, "module") else model

    def _iter_target_modules(self, actor: Any) -> Tuple[torch.nn.Module, ...]:
        if not hasattr(actor, "model_bundle") or actor.model_bundle is None:
            return tuple()
        if not hasattr(actor.model_bundle, "get_no_split_modules"):
            return tuple()

        no_split_modules = actor.model_bundle.get_no_split_modules()
        if not isinstance(no_split_modules, tuple) or not no_split_modules:
            return tuple()

        targets: list[torch.nn.Module] = []
        for _name, module in actor.model.named_modules():
            if isinstance(module, no_split_modules):
                targets.append(module)
        return tuple(targets)

    def _build_state_dict_options(self, **kwargs: Any) -> Any:
        *_, StateDictOptions = self._fsdp2_runtime_apis()

        # API surface changed across torch releases, so keep this tolerant.
        candidate_kwargs = [
            dict(kwargs),
            {k: v for k, v in kwargs.items() if k != "broadcast_from_rank0"},
            {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
            {},
        ]
        for candidate in candidate_kwargs:
            try:
                return StateDictOptions(**candidate)
            except TypeError:
                continue
        # Last-resort attempt, let upstream raise a clear exception.
        return StateDictOptions()

    def wrap_model(self, actor: Any) -> None:
        if not self._use_fsdp:
            logger.info("Rank %s: train_backend=fsdp in no-wrap mode (use_fsdp=false)", actor.rank)
            return

        fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, *_ = self._fsdp2_runtime_apis()

        fsdp_kwargs: Dict[str, Any] = {"reshard_after_forward": True}

        if self._fsdp_config.get("mixed_precision", True):
            param_dtype = parse_torch_dtype(
                self._fsdp_config.get("param_dtype", "bf16") or "fp32",
                field_name="train_backend_kwargs.param_dtype",
            )
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
            )

        if self._fsdp_config.get("cpu_offload", False):
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

        target_modules = self._iter_target_modules(actor)
        for module in target_modules:
            fully_shard(module, **fsdp_kwargs)

        fully_shard(actor.model, **fsdp_kwargs)
        logger.info(
            "Rank %s: Model wrapped with FSDP2 fully_shard (target_modules=%s)",
            actor.rank,
            len(target_modules),
        )

    def _get_full_state_dict(self, actor: Any, *, cpu_offload: bool = True) -> Dict[str, Any]:
        *_a, get_model_state_dict, _b, _c = self._fsdp2_runtime_apis()
        options = self._build_state_dict_options(
            full_state_dict=True,
            cpu_offload=bool(cpu_offload),
        )
        try:
            return dict(get_model_state_dict(actor.model, options=options))
        except TypeError:
            return dict(get_model_state_dict(actor.model))

    def get_state_dict(
        self,
        actor: Any,
        *,
        lora_only: bool = False,
        rank0_only: bool = True,
    ) -> Dict[str, Any]:
        if not self._use_fsdp:
            if rank0_only and actor.rank != 0:
                return {}
            model = self._unwrap_model(actor.model)
            if lora_only:
                peft_lora_state = self._extract_peft_lora_state(model)
                if peft_lora_state:
                    return self._to_cpu_state_dict(peft_lora_state)
                lora_state = self._filter_lora_state(model.state_dict())
                if lora_state:
                    return self._to_cpu_state_dict(lora_state)
                raise ValueError(
                    "LoRA-only sync requested but no LoRA parameters were found in the model state."
                )
            return self._to_cpu_state_dict(model.state_dict())

        full_state_dict = self._get_full_state_dict(actor, cpu_offload=True)
        if rank0_only and actor.rank != 0:
            return {}

        full_state_dict = self._to_cpu_state_dict(full_state_dict)

        if lora_only:
            lora_state = self._filter_lora_state(full_state_dict)
            if lora_state:
                return lora_state
            raise ValueError(
                "LoRA-only sync requested but no LoRA parameters were found in the FSDP state dict."
            )

        return full_state_dict

    def load_state_dict(self, actor: Any, state_dict: Dict[str, Any]) -> None:
        if not self._use_fsdp:
            model = self._unwrap_model(actor.model)
            model.load_state_dict(state_dict)
            return

        *_a, _b, _c, _d, set_model_state_dict, _e = self._fsdp2_runtime_apis()
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
        if self._use_fsdp:
            # FSDP2 collectives already synchronize shards during optimizer steps.
            return
        import torch.distributed as dist

        model = self._unwrap_model(actor.model)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def clip_grad_norm(
        self,
        actor: Any,
        *,
        model: Any,
        max_grad_norm: float,
    ) -> Any:
        def _global_clip_for_sharded_grads() -> torch.Tensor:
            import torch.distributed as dist

            grads = []
            local_sq_sum = 0.0
            for param in model.parameters():
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

        if self._use_fsdp:
            cpu_offload = bool(self._fsdp_config.get("cpu_offload", False))
            if cpu_offload:
                # Use explicit global-norm clipping path for FSDP2+CPU offload.
                # This avoids DTensor CPU collective limitations in clip_grad_norm_.
                return _global_clip_for_sharded_grads()
            try:
                clip_fn = getattr(model, "clip_grad_norm_", None)
                if callable(clip_fn):
                    grad_norm = clip_fn(max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                    )
                return self._maybe_dtensor_to_tensor(grad_norm)
            except RuntimeError as exc:
                # FSDP2 + CPU offload can surface DTensor CPU collective errors here.
                if "No backend type associated with device type cpu" not in str(exc):
                    raise
                logger.warning(
                    "Rank %s: FSDP grad clipping hit CPU DTensor backend error; "
                    "falling back to explicit global-norm clipping path.",
                    getattr(actor, "rank", "unknown"),
                )
                return _global_clip_for_sharded_grads()
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    def topology(self, actor: Any) -> TrainTopology:
        world_size = int(getattr(actor, "world_size", 1))
        if self._use_fsdp:
            return TrainTopology(
                world_size=world_size,
                dp_size=world_size,
                dp_replicate_size=1,
                dp_shard_size=world_size,
                tp_size=1,
                pp_size=1,
                sp_size=1,
                ep_size=1,
                data_partition_axis="dp",
            )
        return TrainTopology(
            world_size=world_size,
            dp_size=world_size,
            dp_replicate_size=world_size,
            dp_shard_size=1,
            tp_size=1,
            pp_size=1,
            sp_size=1,
            ep_size=1,
            data_partition_axis="dp",
        )


__all__ = [
    "FSDPTrainBackendConfig",
    "FSDPTrainBackend",
]
