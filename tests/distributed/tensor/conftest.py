from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
import torch

from diffusionrl.distributed.tensor.batch import (
    Batch,
    concat_field,
    max_field,
    mean_field,
    min_field,
    packed_field,
    shared_field,
    sum_field,
)
from diffusionrl.distributed.tensor.transport import (
    TensorMeta,
    TensorTransport,
    TensorTransportRuntime,
)

# ---------------------------------------------------------------------------
# Test dataclasses — Batch
# ---------------------------------------------------------------------------


@dataclass
class SimpleBatch(Batch):
    data: Optional[torch.Tensor] = concat_field(default=None)
    labels: List[str] = concat_field(default_factory=list)
    config: str = shared_field(default="default")
    wall_clock: float = max_field(default=0.0)


@dataclass
class ReductionBatch(Batch):
    mx: float = max_field(default=0.0)
    mn: float = min_field(default=0.0)
    sm: float = sum_field(default=0.0)
    avg: float = mean_field(default=0.0)


@dataclass
class TupleBatch(Batch):
    items: tuple = concat_field(default_factory=tuple)
    data: Optional[torch.Tensor] = concat_field(default=None)


@dataclass
class DictBatch(Batch):
    mapping: Dict[str, Any] = concat_field(default_factory=dict)


@dataclass
class PackedBatch(Batch):
    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)
    sample_indices: Optional[torch.Tensor] = concat_field(default=None)


@dataclass
class NestedBatch(Batch):
    inner: Optional[SimpleBatch] = concat_field(default=None)
    outer_data: Optional[torch.Tensor] = concat_field(default=None)


# ---------------------------------------------------------------------------
# Test dataclasses for dehydrate/hydrate (plain Batch subclasses)
# ---------------------------------------------------------------------------


@dataclass
class TensorBatch(Batch):
    data: Optional[torch.Tensor] = concat_field(default=None)
    labels: List[str] = concat_field(default_factory=list)
    config: str = shared_field(default="default")


@dataclass
class NestedTensorBatch(Batch):
    child: Optional[TensorBatch] = concat_field(default=None)
    other_data: Optional[torch.Tensor] = concat_field(default=None)


@dataclass
class DictTensorBatch(Batch):
    children: Dict[str, TensorBatch] = concat_field(default_factory=dict)
    other_data: Optional[torch.Tensor] = concat_field(default=None)


@dataclass
class ListTensorBatch(Batch):
    children: List[TensorBatch] = concat_field(default_factory=list)
    other_data: Optional[torch.Tensor] = concat_field(default=None)


# ---------------------------------------------------------------------------
# Mock backend — InMemoryTransport (CPU-only)
# ---------------------------------------------------------------------------


class InMemoryTransport(TensorTransport):
    """CPU-only mock backend storing tensors in a dict keyed by int counter."""

    def __init__(self):
        self._store: Dict[int, torch.Tensor] = {}
        self._counter = 0

    def put(self, tensor: torch.Tensor) -> Any:
        key = self._counter
        self._counter += 1
        self._store[key] = tensor.detach().clone()
        return key

    def get(self, refs: List[Any]) -> torch.Tensor:
        if not refs:
            raise ValueError("empty refs")
        parts = [self._store[key] for key in refs]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorMeta)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_transport_runtime():
    TensorTransportRuntime.clear_current()
    yield
    TensorTransportRuntime.clear_current()


@pytest.fixture
def in_memory_transport():
    return InMemoryTransport()


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
