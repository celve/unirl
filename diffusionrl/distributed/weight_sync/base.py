"""Shared weight-sync primitives: abstract handler bases."""

from __future__ import annotations

import abc
from typing import Any, Tuple

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
    def update_weights(self) -> None:
        raise NotImplementedError


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
    ) -> None:
        super().__init__(model=model, rollout_runtime=rollout_runtime, placement_cfg=placement_cfg)
        self._update_weight_buffer_size = int(bucket_size) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._target_modules = list(target_modules)

    def update_weights(self) -> None:
        self.weight_version += 1
        bucket = []
        bucket_size = 0
        for name, param in raw_state_dict(self.model):
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
