"""Handler-based FSDP weight synchronization helpers."""

from __future__ import annotations

import abc
import logging
import os
import socket
import time
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor

from diffusionrl.distributed.weight_sync_checkpoint import (
    cleanup_published_checkpoint,
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)
from diffusionrl.utils.distributed_utils import init_process_group
from diffusionrl.utils.peft_merge import merged_state_dict, raw_state_dict

logger = logging.getLogger(__name__)


class UpdateWeight(abc.ABC):
    """Base class for weight synchronization handlers."""

    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self) -> None:
        raise NotImplementedError


class BucketedUpdateWeight(UpdateWeight):
    """Weight sync via size-bounded buckets over model state_dict."""

    def update_weights(self) -> None:
        self.weight_version += 1
        bucket = []
        bucket_size = 0
        for name, param in raw_state_dict(self.model):
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket, is_last_bucket=False)
                bucket = []
                bucket_size = 0

            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket, is_last_bucket=True)

    def wait_and_update_bucket_weights(self, bucket, is_last_bucket: bool = False) -> None:
        self.update_bucket_weights(
            bucket,
            weight_version=self.weight_version,
            is_last_bucket=is_last_bucket,
        )

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
        raise NotImplementedError


class UpdateWeightFromTensor(BucketedUpdateWeight):
    """Push model weights to rollout engines via serialized tensors."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        del rollout_engine_lock
        self.rollout_engines = rollout_engines
        self._ipc_gather_src = None
        self._ipc_gather_group = None
        self._ipc_engine = None
        self.tp_rank = 0

        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
        del weight_version
        if self._ipc_gather_group is None or self._ipc_engine is None or self._ipc_gather_src is None:
            return
        try:
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
        except ImportError:
            from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
        from sglang.srt.utils import MultiprocessingSerializer
        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
        except ImportError:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

        monkey_patch_torch_reductions()

        named_tensors_by_dtype = {}
        for name, tensor in named_tensors:
            named_tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        serialized_tensors = []
        for grouped_named_tensors in named_tensors_by_dtype.values():
            bucket = FlattenedTensorBucket(named_tensors=grouped_named_tensors)
            payload = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(payload, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            num_dtypes = len(gathered_serialized_batches[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": (self.args.flush_cache and is_last_bucket and i == num_dtypes - 1),
                    "target_modules": self.args.target_modules,
                }
                ray.get(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))


class UpdateWeightFromDistributed(BucketedUpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = "slime"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = self.args.rollout_num_gpus + 1

            refs = [
                engine.init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size=world_size,
                    group_name=self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
        del weight_version
        if not self._is_src_rank or not named_tensors:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[str(param.dtype) for _, param in named_tensors],
                shapes=[list(param.shape) for _, param in named_tensors],
                group_name=self._group_name,
                target_modules=self.args.target_modules,
                flush_cache=(self.args.flush_cache and is_last_bucket),
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


class UpdateWeightFromCheckpoint(UpdateWeight):
    """Sync weights via filesystem checkpoints."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        del rollout_engines, rollout_engine_lock

    def update_weights(self) -> None:
        self.weight_version += 1
        is_rank0 = dist.get_rank() == 0

        state_dict = {}
        for name, param in merged_state_dict(self.model):
            if is_rank0:
                state_dict[name] = param

        if not is_rank0:
            return

        weight_sync_dir = str(self.args.weight_sync_dir)
        os.makedirs(weight_sync_dir, exist_ok=True)
        path = os.path.join(
            weight_sync_dir,
            f"weights_v{self.weight_version}_{int(time.time_ns())}",
        )

        export_format = str(getattr(self.args, "export_format", "state_dict"))
        if export_format == "sglang_transformer_safetensors":
            publish_sglang_transformer_checkpoint_atomic(state_dict, path, module_name="transformer")
        else:
            publish_checkpoint_atomic(state_dict, path)

        rollout_manager = self.args.rollout_manager
        update_fn = getattr(rollout_manager, "update_weights_from_path", None)
        if update_fn is None:
            raise RuntimeError("rollout_manager does not expose update_weights_from_path().")
        remote_fn = getattr(update_fn, "remote", None)
        if callable(remote_fn):
            ray.get(remote_fn(path))
        else:
            update_fn(path)
        cleanup_published_checkpoint(path)
