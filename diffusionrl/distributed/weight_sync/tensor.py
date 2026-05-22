"""Tensor-payload weight sync: ship serialized named tensors over Ray IPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import ray
import torch.distributed as dist

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.distributed.weight_sync.base import BucketedUpdateWeight
from diffusionrl.utils.peft_merge import extract_lora_tensors


@register_config(
    group="sync",
    name="tensor_payload",
    target="diffusionrl.distributed.weight_sync.tensor.UpdateWeightFromTensor",
    expand=True,
)
@dataclass(frozen=True)
class TensorPayloadSyncConfig:
    """Push model weights to sglang rollout engines via serialized tensors."""

    bucket_size: int = 256
    flush_cache: bool = True
    target_modules: Tuple[str, ...] = field(default_factory=lambda: ("transformer",))

    def __post_init__(self) -> None:
        require(self.bucket_size >= 1, f"TensorPayloadSyncConfig.bucket_size must be >= 1; got {self.bucket_size!r}")
        require(
            len(self.target_modules) >= 1,
            f"TensorPayloadSyncConfig.target_modules must not be empty; got {self.target_modules!r}",
        )
        require(
            all(str(name or "").strip() for name in self.target_modules),
            f"TensorPayloadSyncConfig.target_modules cannot contain empty entries; got {self.target_modules!r}",
        )


class UpdateWeightFromTensor(BucketedUpdateWeight):
    """Push model weights to rollout engines via serialized tensors."""

    def update_weights(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ) -> None:
        if peft_config and base_sync_done:
            # All FSDP ranks must materialize the LoRA DTensors so the
            # underlying all_gathers complete; only the local gather source
            # sends the resulting tensors to its rollout actor.
            lora_tensors = extract_lora_tensors(
                self.model,
                param_name_prefix=self._param_name_prefix,
            )
            if self._ipc_engine is None or self._ipc_gather_src is None:
                return
            if dist.get_rank() != self._ipc_gather_src:
                return
            ray.get(
                self._ipc_engine.set_lora_from_tensors.remote(
                    "default",
                    lora_tensors,
                    peft_config=peft_config,
                )
            )
            return

        super().update_weights()

    def connect_rollout_engines(self) -> None:
        rollout_engines = list(self._rollout_runtime.get_rollout_actors())
        self.rollout_engines = rollout_engines
        num_gpus_per_engine = int(self._placement_cfg.num_rollout_gpus_per_actor)
        self._ipc_gather_src = None
        self._ipc_gather_group = None
        self._ipc_engine = None
        self.tp_rank = 0

        for i, engine in enumerate(rollout_engines):
            start_rank = i * num_gpus_per_engine
            end_rank = (i + 1) * num_gpus_per_engine
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
                    "flush_cache": (self._flush_cache and is_last_bucket and i == num_dtypes - 1),
                    "target_modules": self._target_modules,
                }
                ray.get(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))


__all__ = ["TensorPayloadSyncConfig", "UpdateWeightFromTensor"]
