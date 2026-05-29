"""Filesystem-checkpoint weight sync: publish a checkpoint, point rollout at it."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import ray
import torch.distributed as dist

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.distributed.weight_sync.base import UpdateWeight
from diffusionrl.distributed.weight_sync_checkpoint import (
    cleanup_published_checkpoint,
    publish_checkpoint_atomic,
    publish_sglang_transformer_checkpoint_atomic,
)
from diffusionrl.utils.peft_merge import merged_state_dict

_VALID_EXPORT_FORMATS = frozenset({"state_dict", "sglang_transformer_safetensors"})


@register_config(
    group="sync",
    name="checkpoint_path",
    target="diffusionrl.distributed.weight_sync.checkpoint.UpdateWeightFromCheckpoint",
    expand=True,
)
@dataclass(frozen=True)
class CheckpointSyncConfig:
    """Sync weights via filesystem checkpoint; rollout loads from the path."""

    dir: str = "/tmp/diffusionrl/weight_sync"
    export_format: str = "state_dict"

    def __post_init__(self) -> None:
        require(bool(str(self.dir).strip()), f"CheckpointSyncConfig.dir must be a non-empty path; got {self.dir!r}")
        require(
            self.export_format in _VALID_EXPORT_FORMATS,
            f"CheckpointSyncConfig.export_format must be one of {sorted(_VALID_EXPORT_FORMATS)}; got {self.export_format!r}",
        )


class UpdateWeightFromCheckpoint(UpdateWeight):
    """Sync weights via filesystem checkpoints."""

    def __init__(
        self,
        *,
        model,
        rollout_runtime,
        placement_cfg,
        dir: str,
        export_format: str,
    ) -> None:
        super().__init__(model=model, rollout_runtime=rollout_runtime, placement_cfg=placement_cfg)
        self._weight_sync_dir = str(dir)
        self._export_format = str(export_format)

    def connect_rollout_engines(self) -> None:
        return

    def update_weights(
        self,
        *,
        model: object | None = None,
        peft_config: dict | None = None,
        base_sync_done: bool = False,
        param_name_prefix: str | None = None,
        packed_modules: dict | None = None,
        track_prefix: str = "",
        use_merged: bool = False,
    ) -> None:
        del peft_config, base_sync_done, param_name_prefix, packed_modules, track_prefix, use_merged
        resolved_model = self._resolve_model(model)
        self.weight_version += 1
        is_rank0 = dist.get_rank() == 0

        state_dict = {}
        for name, param in merged_state_dict(resolved_model):
            if is_rank0:
                state_dict[name] = param

        if not is_rank0:
            return

        os.makedirs(self._weight_sync_dir, exist_ok=True)
        path = os.path.join(
            self._weight_sync_dir,
            f"weights_v{self.weight_version}_{int(time.time_ns())}",
        )

        if self._export_format == "sglang_transformer_safetensors":
            publish_sglang_transformer_checkpoint_atomic(state_dict, path, module_name="transformer")
        else:
            publish_checkpoint_atomic(state_dict, path)

        update_fn = getattr(self._rollout_runtime, "update_weights_from_path", None)
        if update_fn is None:
            raise RuntimeError("rollout_runtime does not expose update_weights_from_path().")
        remote_fn = getattr(update_fn, "remote", None)
        if callable(remote_fn):
            ray.get(remote_fn(path))
        else:
            update_fn(path)
        cleanup_published_checkpoint(path)


__all__ = ["CheckpointSyncConfig", "UpdateWeightFromCheckpoint"]
