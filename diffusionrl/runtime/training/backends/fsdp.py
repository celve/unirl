"""FSDP training backend implementation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .base import TrainBackend, TrainBackendCapabilities

logger = logging.getLogger(__name__)


class FSDPTrainBackend(TrainBackend):
    """FSDP backend with optional no-wrap mode for local debugging."""

    BACKEND_NAME = "fsdp"

    def __init__(self, *, backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
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
            notes="Default backend. Supports FSDP full-state export and direct-sampling mode.",
        )

    def uses_sharded_model(self) -> bool:
        return bool(self._use_fsdp)

    def before_model_load(self, actor: Any) -> None:
        actor._fsdp_cpu_offload = bool(self._fsdp_config.get("cpu_offload", False) and self._use_fsdp)

    def wrap_model(self, actor: Any) -> None:
        if not self._use_fsdp:
            logger.info("Rank %s: train_backend=fsdp in no-wrap mode (use_fsdp=false)", actor.rank)
            return

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = strategy_map.get(
            self._fsdp_config.get("sharding_strategy", "FULL_SHARD"),
            ShardingStrategy.FULL_SHARD,
        )

        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map.get(
            self._fsdp_config.get("backward_prefetch", "BACKWARD_PRE"),
            BackwardPrefetch.BACKWARD_PRE,
        )

        cpu_offload = None
        if self._fsdp_config.get("cpu_offload", False):
            cpu_offload = CPUOffload(offload_params=True)

        mixed_precision = None
        if self._fsdp_config.get("mixed_precision", True):
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        auto_wrap_policy = None
        if hasattr(actor.model_bundle, "get_no_split_modules"):
            no_split_modules = actor.model_bundle.get_no_split_modules()
            if no_split_modules:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=no_split_modules,
                )

        actor.model = FSDP(
            actor.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=actor._device,
            use_orig_params=True,
        )
        logger.info("Rank %s: Model wrapped with FSDP", actor.rank)

    def _state_dict_context(self, actor: Any, *, rank0_only: bool):
        if not self._use_fsdp:
            return nullcontext()
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        return FSDP.state_dict_type(actor.model, StateDictType.FULL_STATE_DICT, save_policy)

    @staticmethod
    def _to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() if hasattr(v, "is_cuda") and v.is_cuda else v for k, v in state_dict.items()}

    @staticmethod
    def _filter_lora_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}

    @staticmethod
    def _extract_peft_lora_state(model: Any) -> Dict[str, torch.Tensor]:
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

        lora_state: Dict[str, torch.Tensor] = {}
        for adapter_name in adapter_names:
            lora_state.update(get_peft_model_state_dict(base_model, adapter_name=adapter_name))
        return lora_state

    def get_state_dict(
        self,
        actor: Any,
        *,
        lora_only: bool = False,
        rank0_only: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if lora_only:
            try:
                if self._use_fsdp:
                    with self._state_dict_context(actor, rank0_only=rank0_only):
                        full_state_dict = actor.model.state_dict()
                    if rank0_only and actor.rank != 0:
                        return {}
                    lora_state = self._filter_lora_state(full_state_dict)
                    if lora_state:
                        return self._to_cpu_state_dict(lora_state)
                    logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")
                    return self._to_cpu_state_dict(full_state_dict)

                peft_lora_state = self._extract_peft_lora_state(actor.model)
                if peft_lora_state:
                    return self._to_cpu_state_dict(peft_lora_state)

                local_state = actor.model.state_dict()
                lora_state = self._filter_lora_state(local_state)
                if lora_state:
                    return self._to_cpu_state_dict(lora_state)
                logger.warning("LoRA-only sync found no LoRA keys; falling back to full state_dict.")
            except Exception as exc:
                logger.warning("LoRA-only sync failed; falling back to full sync: %s", exc)

        if self._use_fsdp:
            with self._state_dict_context(actor, rank0_only=rank0_only):
                state_dict = actor.model.state_dict()
            if rank0_only and actor.rank != 0:
                return {}
            return self._to_cpu_state_dict(state_dict)

        if rank0_only and actor.rank != 0:
            return {}
        return self._to_cpu_state_dict(actor.model.state_dict())

    def load_state_dict(self, actor: Any, state_dict: Dict[str, torch.Tensor]) -> None:
        if self._use_fsdp:
            with self._state_dict_context(actor, rank0_only=True):
                actor.model.load_state_dict(state_dict)
            return
        actor.model.load_state_dict(state_dict)

    def broadcast_parameters(self, actor: Any) -> None:
        if self._use_fsdp:
            return
        for param in actor.model.parameters():
            dist.broadcast(param.data, src=0)

    def buffer_consumer_spec(self, actor: Any) -> Dict[str, Any]:
        dp_size = self.data_parallel_size(actor)
        return {
            "dp_size": dp_size,
            "partition_train_data": True,
            "partition_mode": "data_parallel",
        }


__all__ = ["FSDPTrainBackend"]
