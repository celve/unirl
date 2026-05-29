"""TensorStoreTransport: worker-local GPU tensor registry backend."""

from __future__ import annotations

from typing import Any, List

import torch

from diffusionrl.distributed.tensor.transport import TensorMeta, TensorTransport


class TensorStoreTransport(TensorTransport):
    """TensorStore backend — per-tensor put/get with IPC handles."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def put(self, tensor: torch.Tensor) -> Any:
        return self._store.put(tensor)

    def get(self, refs: List[Any]) -> torch.Tensor:
        if not refs:
            raise ValueError("TensorStoreTransport.get: empty refs list")
        parts = [self._store.get(h) for h in refs]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorMeta)


__all__ = ["TensorStoreTransport"]
