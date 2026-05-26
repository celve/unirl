"""Shared weight-sync primitives: abstract handler bases."""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import torch.nn as nn

from diffusionrl.ray.placement import PlacementConfig
from diffusionrl.utils.peft_merge import raw_state_dict


class UpdateWeight(abc.ABC):
    """Base class for weight-synchronization handlers."""

    def __init__(
        self,
        *,
        model: nn.Module,
        rollout_runtime: Any,
        placement_cfg: PlacementConfig,
    ) -> None:
        self.model = model
        self._rollout_runtime = rollout_runtime
        self._placement_cfg = placement_cfg
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(
        self,
        *,
        model: Optional[nn.Module] = None,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        param_name_prefix: Optional[str] = None,
        packed_modules: Optional[dict] = None,
        track_prefix: str = "",
    ) -> None:
        raise NotImplementedError

    def _resolve_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        return model if model is not None else self.model

    def _resolve_prefix(self, param_name_prefix: Optional[str] = None) -> str:
        if param_name_prefix is not None:
            return param_name_prefix
        return getattr(self, "_param_name_prefix", "")

    def _apply_track_prefix(self, name: str, track_prefix: str) -> str:
        if track_prefix:
            return f"{track_prefix}.{name}"
        return name


class BucketedUpdateWeight(UpdateWeight):
    """Weight sync via size-bounded buckets over the model state_dict."""

    def __init__(
        self,
        *,
        model: nn.Module,
        rollout_runtime: Any,
        placement_cfg: PlacementConfig,
        bucket_size: int,
        flush_cache: bool,
        target_modules: Tuple[str, ...],
        param_name_prefix: str = "",
    ) -> None:
        super().__init__(model=model, rollout_runtime=rollout_runtime, placement_cfg=placement_cfg)
        self._update_weight_buffer_size = int(bucket_size) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._target_modules = list(target_modules)
        self._param_name_prefix = str(param_name_prefix or "")

    def update_weights(
        self,
        *,
        model: Optional[nn.Module] = None,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        param_name_prefix: Optional[str] = None,
        packed_modules: Optional[dict] = None,
        track_prefix: str = "",
    ) -> None:
        del peft_config, base_sync_done, packed_modules
        resolved_model = self._resolve_model(model)
        prefix = self._resolve_prefix(param_name_prefix)
        self.weight_version += 1
        bucket = []
        bucket_size = 0
        for name, param in raw_state_dict(resolved_model):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            if prefix:
                name = prefix + name
            name = self._apply_track_prefix(name, track_prefix)
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self._update_weight_buffer_size:
                self._flush_bucket(bucket, is_last_bucket=False)
                bucket = []
                bucket_size = 0

            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            self._flush_bucket(bucket, is_last_bucket=True)

    def _flush_bucket(self, bucket, is_last_bucket: bool = False) -> None:
        self.update_bucket_weights(
            bucket,
            weight_version=self.weight_version,
            is_last_bucket=is_last_bucket,
        )

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
        raise NotImplementedError


__all__ = ["UpdateWeight", "BucketedUpdateWeight"]
