"""Storage backends for buffer-owned training batches."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class BatchStore(Protocol):
    """Minimal storage interface used by the buffer subsystem."""

    def put(self, batch: Any) -> Any: ...

    def get(self, handle: Any) -> Any: ...

    def release(self, handle: Any) -> None: ...


class InMemoryBatchStore:
    """Test-friendly batch store that retains batches in local process memory."""

    def __init__(self) -> None:
        self._next_handle = 0
        self._batches: Dict[str, Any] = {}

    def put(self, batch: Any) -> str:
        self._next_handle += 1
        handle = f"batch:{self._next_handle}"
        self._batches[handle] = batch
        return handle

    def get(self, handle: Any) -> Any:
        return self._batches[handle]

    def release(self, handle: Any) -> None:
        self._batches.pop(str(handle), None)

    def live_ref_count(self) -> int:
        return len(self._batches)


class RayBatchStore:
    """Object-store-backed batch storage used by the Ray actor shell."""

    def put(self, batch: Any) -> Any:
        import ray

        return ray.put(batch)

    def get(self, handle: Any) -> Any:
        import ray

        return ray.get(handle)

    def release(self, handle: Any) -> None:
        del handle
        return None


__all__ = [
    "BatchStore",
    "InMemoryBatchStore",
    "RayBatchStore",
]
