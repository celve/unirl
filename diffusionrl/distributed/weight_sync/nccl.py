"""NCCL-broadcast weight sync: temporary NCCL group, broadcast params."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Optional, Tuple

import ray
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.distributed.weight_sync.base import BucketedUpdateWeight
from diffusionrl.utils.distributed_utils import init_process_group
from diffusionrl.utils.peft_merge import extract_lora_tensors


@register_config(
    group="sync",
    name="nccl_broadcast",
    target="diffusionrl.distributed.weight_sync.nccl.UpdateWeightFromDistributed",
    expand=True,
)
@dataclass(frozen=True)
class NcclBroadcastSyncConfig:
    """Broadcast model weights to rollout engines via a temporary NCCL group."""

    bucket_size: int = 256
    flush_cache: bool = True
    target_modules: Tuple[str, ...] = field(default_factory=lambda: ("transformer",))

    def __post_init__(self) -> None:
        require(self.bucket_size >= 1, f"NcclBroadcastSyncConfig.bucket_size must be >= 1; got {self.bucket_size!r}")
        require(
            len(self.target_modules) >= 1,
            f"NcclBroadcastSyncConfig.target_modules must not be empty; got {self.target_modules!r}",
        )
        require(
            all(str(name or "").strip() for name in self.target_modules),
            f"NcclBroadcastSyncConfig.target_modules cannot contain empty entries; got {self.target_modules!r}",
        )


class UpdateWeightFromDistributed(BucketedUpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def __init__(
        self,
        *,
        model,
        rollout_runtime,
        placement_cfg,
        bucket_size: int,
        flush_cache: bool,
        target_modules: Tuple[str, ...],
        param_name_prefix: str = "",
    ) -> None:
        super().__init__(
            model=model,
            rollout_runtime=rollout_runtime,
            placement_cfg=placement_cfg,
            bucket_size=bucket_size,
            flush_cache=flush_cache,
            target_modules=target_modules,
            param_name_prefix=param_name_prefix,
        )

    def update_weights(
        self,
        *,
        model: Optional[object] = None,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        param_name_prefix: Optional[str] = None,
        packed_modules: Optional[dict] = None,
        track_prefix: str = "",
        use_merged: bool = False,
    ) -> None:
        if track_prefix:
            raise NotImplementedError(
                "nccl_broadcast does not support multi-track composed routing. "
                "Use sync=tensor_payload so each track is sent to the matching rollout child."
            )
        resolved_model = self._resolve_model(model)
        resolved_prefix = self._resolve_prefix(param_name_prefix)
        if peft_config and base_sync_done:
            lora_tensors = extract_lora_tensors(
                resolved_model,
                param_name_prefix=resolved_prefix,
                packed_modules=packed_modules if packed_modules is not None else getattr(self, "_packed_modules", None),
            )
            if track_prefix:
                lora_tensors = {f"{track_prefix}.{k}": v for k, v in lora_tensors.items()}
            if not self._is_src_rank:
                return
            refs = [
                actor.set_lora_from_tensors.remote(
                    "default",
                    lora_tensors,
                    peft_config=peft_config,
                )
                for actor in self.rollout_engines
            ]
            ray.get(refs)
            return

        super().update_weights(
            model=model,
            param_name_prefix=param_name_prefix,
            packed_modules=packed_modules,
            track_prefix=track_prefix,
            use_merged=use_merged,
        )

    def connect_rollout_engines(self) -> None:
        rollout_engines = list(self._rollout_runtime.get_rollout_actors())
        self.rollout_engines = rollout_engines
        num_gpus_per_engine = int(self._placement_cfg.num_rollout_gpus_per_actor)
        total_rollout_gpus = int(self._placement_cfg.total_rollout_gpus)
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = "slime"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = total_rollout_gpus + 1

            refs = [
                engine.init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=i * num_gpus_per_engine + 1,
                    world_size=world_size,
                    group_name=self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(
        self, named_tensors, weight_version=None, is_last_bucket: bool = False, track_prefix: str = ""
    ) -> None:
        del weight_version
        if not self._is_src_rank or not named_tensors:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[str(param.dtype) for _, param in named_tensors],
                shapes=[list(param.shape) for _, param in named_tensors],
                group_name=self._group_name,
                target_modules=self._target_modules,
                flush_cache=(self._flush_cache and is_last_bucket),
            )
            for engine in self.rollout_engines
        ]

        handles = []
        for _name, param in named_tensors:
            param_data = param.data.contiguous()
            if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                param_data = param_data.full_tensor()
            handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

        for handle in handles:
            handle.wait()
        ray.get(refs)


__all__ = ["NcclBroadcastSyncConfig", "UpdateWeightFromDistributed"]
