"""FSDP2 training backend implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .base import TrainBackend, TrainBackendCapabilities, TrainTopology

logger = logging.getLogger(__name__)


def _safe_dtype(name: str) -> torch.dtype:
    key = str(name or "bf16").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported FSDP2 param_dtype={name!r}. Use one of bf16/fp16/fp32.")


class FSDPTrainBackend(TrainBackend):
    """FSDP2 backend (composable fully_shard).

    Notes:
    - Requires a torch build exposing `torch.distributed.fsdp.fully_shard`
      and distributed checkpoint state-dict helpers.
    - `use_fsdp=false` still keeps no-wrap debug behavior for local validation.
    """

    BACKEND_NAME = "fsdp"

    def __init__(self, *, backend_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(backend_kwargs=backend_kwargs)
        kwargs = dict(self.backend_kwargs)
        fsdp_config = kwargs.pop("fsdp_config", None)

        merged: Dict[str, Any] = {}
        if isinstance(fsdp_config, dict):
            merged.update(fsdp_config)
        merged.update(kwargs)

        self._use_fsdp = bool(merged.pop("use_fsdp", True))
        self._fsdp_config = merged

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
            preferred_weight_transport="checkpoint_path",
            preferred_weight_export_format="state_dict",
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
            from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                set_model_state_dict,
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

        sharding_strategy = str(self._fsdp_config.get("sharding_strategy", "FULL_SHARD") or "FULL_SHARD").upper()
        if sharding_strategy != "FULL_SHARD":
            logger.warning(
                "Rank %s: FSDP2 backend ignores sharding_strategy=%s. "
                "Current implementation is fully_shard-only.",
                actor.rank,
                sharding_strategy,
            )
        backward_prefetch = str(self._fsdp_config.get("backward_prefetch", "BACKWARD_PRE") or "BACKWARD_PRE").upper()
        if backward_prefetch != "BACKWARD_PRE":
            logger.warning(
                "Rank %s: FSDP2 backend ignores backward_prefetch=%s.",
                actor.rank,
                backward_prefetch,
            )

        fsdp_kwargs: Dict[str, Any] = {}

        if self._fsdp_config.get("mixed_precision", True):
            param_dtype = _safe_dtype(self._fsdp_config.get("param_dtype", "bf16"))
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
                try:
                    peft_lora_state = self._extract_peft_lora_state(model)
                    if peft_lora_state:
                        return self._to_cpu_state_dict(peft_lora_state)
                    lora_state = self._filter_lora_state(model.state_dict())
                    if lora_state:
                        return self._to_cpu_state_dict(lora_state)
                    logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")
                except Exception as exc:
                    logger.warning("LoRA-only sync failed; falling back to full sync: %s", exc)
            return self._to_cpu_state_dict(model.state_dict())

        full_state_dict = self._get_full_state_dict(actor, cpu_offload=True)
        if rank0_only and actor.rank != 0:
            return {}

        full_state_dict = self._to_cpu_state_dict(full_state_dict)

        if lora_only:
            try:
                lora_state = self._filter_lora_state(full_state_dict)
                if lora_state:
                    return lora_state
                logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")
            except Exception as exc:
                logger.warning("LoRA-only sync failed; falling back to full sync: %s", exc)

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
        if self._use_fsdp:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            return self._maybe_dtensor_to_tensor(grad_norm)
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


__all__ = ["FSDPTrainBackend"]
