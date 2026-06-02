"""TQTransport: Transfer Queue backend for TensorTransport."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from tensordict import NonTensorData, TensorDict

from unirl.distributed.tensor.backend.transfer_queue.runtime import (
    _DEFAULT_PARTITION_ID,
    TransferQueueRuntime,
    _run_async_in_temp_loop,
)
from unirl.distributed.tensor.transport import (
    TensorMeta,
    TensorTransport,
)


class TQTransport(TensorTransport):
    """Transfer Queue backend — batches named tensors into single round-trips."""

    def __init__(
        self,
        runtime: TransferQueueRuntime,
        partition_id: str = _DEFAULT_PARTITION_ID,
    ) -> None:
        self._runtime = runtime
        self._partition_id = partition_id

    @property
    def _client(self) -> Any:
        return self._runtime.client

    def put(self, tensor: torch.Tensor) -> Any:
        async def _put() -> Any:
            bs = int(tensor.shape[0]) if tensor.dim() > 0 else 1
            td = TensorDict({"_": tensor}, batch_size=bs).cpu()
            return await self._client.async_put(data=td, partition_id=self._partition_id)

        return _run_async_in_temp_loop(_put)

    def get(self, refs: List[Any]) -> torch.Tensor:
        async def _get() -> torch.Tensor:
            union = refs[0]
            for m in refs[1:]:
                union = union.union(m)
            td = await self._client.async_get_data(union)
            keys = list(td.keys())
            return td[keys[0]]

        return _run_async_in_temp_loop(_get)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorMeta)

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorMeta]:
        if not tensors:
            return {}

        async def _put_batch() -> Dict[str, TensorMeta]:
            first = next(iter(tensors.values()))
            bs = int(first.shape[0]) if first.dim() > 0 else 1
            d: dict = {}
            for k, t in tensors.items():
                if isinstance(t, torch.Tensor):
                    d[k] = t
                elif isinstance(t, list):
                    d[k] = NonTensorData(t)
                else:
                    raise TypeError(f"TQTransport.put_batch: unsupported type {type(t).__name__} for key {k!r}")
            td = TensorDict(d, batch_size=bs).cpu()
            meta = await self._client.async_put(data=td, partition_id=self._partition_id)

            result: Dict[str, TensorMeta] = {}
            for k, t in tensors.items():
                key_meta = meta.select_fields([k])
                result[k] = TensorMeta(
                    refs=[key_meta],
                    sizes=[bs],
                    shape=tuple(t.shape) if isinstance(t, torch.Tensor) else None,
                    dtype=t.dtype if isinstance(t, torch.Tensor) else None,
                    device=str(t.device) if isinstance(t, torch.Tensor) else None,
                )
            return result

        return _run_async_in_temp_loop(_put_batch)

    def get_batch(self, metas: Dict[str, TensorMeta]) -> Dict[str, torch.Tensor]:
        if not metas:
            return {}

        async def _get_batch() -> Dict[str, torch.Tensor]:
            all_refs = []
            for m in metas.values():
                all_refs.extend(m.refs)
            if not all_refs:
                return {}
            union = all_refs[0]
            for r in all_refs[1:]:
                union = union.union(r)
            td = await self._client.async_get_data(union)

            result: Dict[str, torch.Tensor] = {}
            for k in metas:
                if k in td:
                    val = td[k]
                    if isinstance(val, list) and val and all(isinstance(t, torch.Tensor) for t in val):
                        result[k] = torch.stack(val, dim=0)
                    else:
                        result[k] = val
            return result

        return _run_async_in_temp_loop(_get_batch)


__all__ = ["TQTransport"]
