"""Field-metadata-driven batch container with automatic concat/select/slice.

Annotate dataclass fields with one of the field-kind constructors and the
``Batched`` base class provides generic implementations of concat, select,
slice, to_device, and clone that dispatch on the field kind and value type.

Field kinds:
  - ``concat_field()`` — per-sample, batch-aligned; concatenated along dim 0.
  - ``shared_field()`` — identical across samples; first value taken on concat.
  - ``max_field()`` / ``min_field()`` / ``sum_field()`` / ``mean_field()`` —
    scalar or same-shape-across-instances; reduced across instances on concat
    using the named reduction. Like ``shared_field`` these are not batch-
    aligned and pass through ``select`` / ``slice`` untouched.

Supported value types for concat fields:
  - ``torch.Tensor`` with batch dim at axis 0
  - ``list`` or ``tuple`` with ``len == batch_size``
  - ``dict`` containing tensors / lists / nested dicts (recursive)
  - Nested ``Batched`` instances
  - ``None`` (optional fields)

Example::

    @dataclass
    class MyBatch(Batched):
        data: torch.Tensor = concat_field()
        labels: List[str] = concat_field(default_factory=list)
        schedule: torch.Tensor = shared_field()
        config: str = shared_field(default="default")
        wall_clock: float = max_field(default=0.0)
"""

from __future__ import annotations

import copy
from dataclasses import field, fields as dc_fields
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import torch

T = TypeVar("T", bound="Batched")

_FIELD_KIND_KEY = "field_kind"


class FieldKind(Enum):
    CONCAT = auto()
    SHARED = auto()
    MAX = auto()
    MIN = auto()
    SUM = auto()
    MEAN = auto()


_REDUCTION_KINDS = frozenset(
    {FieldKind.MAX, FieldKind.MIN, FieldKind.SUM, FieldKind.MEAN}
)


# ---------------------------------------------------------------------------
# Field constructors
# ---------------------------------------------------------------------------

def concat_field(**kwargs: Any) -> Any:
    """Declare a per-sample field (concatenated along the batch dimension)."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = FieldKind.CONCAT
    return field(metadata=metadata, **kwargs)


def shared_field(**kwargs: Any) -> Any:
    """Declare a shared field (identical for every sample in the batch)."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = FieldKind.SHARED
    return field(metadata=metadata, **kwargs)


def _reduction_field(kind: FieldKind, **kwargs: Any) -> Any:
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = kind
    return field(metadata=metadata, **kwargs)


def max_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise max on concat."""
    return _reduction_field(FieldKind.MAX, **kwargs)


def min_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise min on concat."""
    return _reduction_field(FieldKind.MIN, **kwargs)


def sum_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise sum on concat."""
    return _reduction_field(FieldKind.SUM, **kwargs)


def mean_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise mean on concat."""
    return _reduction_field(FieldKind.MEAN, **kwargs)


def _field_kind(f: Any) -> FieldKind:
    return f.metadata.get(_FIELD_KIND_KEY, FieldKind.SHARED)


# ---------------------------------------------------------------------------
# Value-level helpers
# ---------------------------------------------------------------------------

def _infer_batch_size(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, Batched):
        return value.batch_size
    if isinstance(value, dict):
        for v in value.values():
            bs = _infer_batch_size(v)
            if bs is not None:
                return bs
    return None


def _reduce_value(values: List[Any], kind: FieldKind) -> Any:
    """Reduce a per-instance value across N instances by ``kind``."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    if all(isinstance(v, torch.Tensor) for v in non_none):
        stacked = torch.stack(non_none, dim=0)
        if kind is FieldKind.MAX:
            return stacked.amax(dim=0)
        if kind is FieldKind.MIN:
            return stacked.amin(dim=0)
        if kind is FieldKind.SUM:
            return stacked.sum(dim=0)
        if kind is FieldKind.MEAN:
            return stacked.mean(dim=0)

    if kind is FieldKind.MAX:
        return max(non_none)
    if kind is FieldKind.MIN:
        return min(non_none)
    if kind is FieldKind.SUM:
        return sum(non_none)
    if kind is FieldKind.MEAN:
        total = sum(non_none)
        return total / len(non_none)

    raise ValueError(f"Unsupported reduction kind: {kind}")


def _concat_value(values: List[Any], batch_sizes: List[int]) -> Any:
    """Concatenate per-sample values along the batch axis."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    if all(isinstance(v, torch.Tensor) for v in non_none):
        is_batched = [
            v.dim() > 0 and int(v.shape[0]) == bs
            for v, bs in zip(values, batch_sizes)
            if v is not None
        ]
        if all(is_batched):
            return torch.cat(non_none, dim=0)
        if not any(is_batched):
            return non_none[0]
        raise ValueError("Mixed batched / non-batched tensors in concat field")

    if all(isinstance(v, list) for v in non_none):
        is_batched = [
            len(v) == bs
            for v, bs in zip(values, batch_sizes)
            if v is not None
        ]
        if all(is_batched):
            merged: List[Any] = []
            for v in non_none:
                merged.extend(v)
            return merged

    if all(isinstance(v, tuple) for v in non_none):
        is_batched = [
            len(v) == bs
            for v, bs in zip(values, batch_sizes)
            if v is not None
        ]
        if all(is_batched):
            merged_t: List[Any] = []
            for v in non_none:
                merged_t.extend(v)
            return tuple(merged_t)

    if all(isinstance(v, dict) for v in non_none):
        keys = sorted({k for v in non_none for k in v})
        return {
            k: _concat_value(
                [v.get(k) if isinstance(v, dict) else None for v in values],
                batch_sizes=batch_sizes,
            )
            for k in keys
        }

    if all(isinstance(v, Batched) for v in non_none):
        return type(non_none[0]).concat(non_none)

    first = non_none[0]
    if torch.is_tensor(first):
        if all(
            torch.is_tensor(v) and v.shape == first.shape and torch.equal(v.to(first.device), first)
            for v in non_none[1:]
        ):
            return first
    elif all(v == first for v in non_none[1:]):
        return copy.deepcopy(first)

    raise ValueError(
        f"Cannot concat values: types={[type(v).__name__ for v in values]}, "
        f"batch_sizes={batch_sizes}"
    )


def _to_index_list(indices: Union[torch.Tensor, Sequence[int]]) -> List[int]:
    if isinstance(indices, torch.Tensor):
        return indices.tolist()
    return list(indices)


def _to_index_tensor(indices: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        return indices
    return torch.tensor(indices, dtype=torch.long)


def _select_value(
    value: Any, indices: Union[torch.Tensor, Sequence[int]], batch_size: int,
) -> Any:
    """Re-index a per-sample value by an index tensor or list of ints."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            idx = _to_index_tensor(indices)
            return value.index_select(0, idx.to(value.device))
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return [value[i] for i in _to_index_list(indices)]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[i] for i in _to_index_list(indices))
    if isinstance(value, dict):
        return {k: _select_value(v, indices, batch_size) for k, v in value.items()}
    if isinstance(value, Batched):
        return value.select(indices)
    return value


def _slice_value(value: Any, start: int, end: int, batch_size: int) -> Any:
    """Slice a per-sample value along the batch dimension."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            return value[start:end].clone()
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return list(value[start:end])
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[start:end])
    if isinstance(value, dict):
        return {k: _slice_value(v, start, end, batch_size) for k, v in value.items()}
    if isinstance(value, Batched):
        return value.slice(start, end)
    return value


def _move_value(value: Any, device: Union[str, torch.device]) -> Any:
    """Move tensors in a value tree to *device*."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_value(v, device) for v in value]
        return type(value)(moved)
    if isinstance(value, Batched):
        return value.to_device(device)
    return value


def _clone_value(value: Any) -> Any:
    """Deep-clone a value (tensors are ``.clone()``'d, dicts/lists recursed)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cloned = [_clone_value(v) for v in value]
        return type(value)(cloned)
    if isinstance(value, Batched):
        return value.clone()
    return copy.deepcopy(value)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Batched:
    """Mixin / base for ``@dataclass`` containers with concat/shared fields.

    Subclasses must be ``@dataclass``es whose fields are annotated with
    ``concat_field()`` or ``shared_field()``.  Fields without annotation are
    treated as shared.

    ``batch_size`` is inferred from the first non-None concat field.
    Subclasses may override the property if custom logic is needed.
    """

    @property
    def batch_size(self) -> int:
        for f in dc_fields(self):  # type: ignore[arg-type]
            if _field_kind(f) is not FieldKind.CONCAT:
                continue
            bs = _infer_batch_size(getattr(self, f.name))
            if bs is not None:
                return bs
        return 0

    @classmethod
    def concat(cls: Type[T], items: Sequence[T]) -> T:
        """Concatenate multiple instances along the batch dimension."""
        if not items:
            raise ValueError(f"Cannot concat empty sequence of {cls.__name__}")
        if len(items) == 1:
            return items[0]

        batch_sizes = [item.batch_size for item in items]
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(items[0]):  # type: ignore[arg-type]
            values = [getattr(item, f.name) for item in items]
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _concat_value(values, batch_sizes)
            elif kind in _REDUCTION_KINDS:
                kwargs[f.name] = _reduce_value(values, kind)
            else:
                kwargs[f.name] = values[0]
        return cls(**kwargs)

    def concat_with(self: T, *others: T) -> T:
        """Concatenate ``self`` with one or more other instances."""
        return type(self).concat([self, *others])

    def select(self: T, indices: torch.Tensor) -> T:
        """Re-index along the batch dimension (gather / shuffle / subsample)."""
        bs = self.batch_size
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if _field_kind(f) is FieldKind.CONCAT:
                kwargs[f.name] = _select_value(val, indices, bs)
            else:
                kwargs[f.name] = val
        return type(self)(**kwargs)

    def slice(self: T, start: int, end: int) -> T:
        """Slice ``[start, end)`` along the batch dimension."""
        bs = self.batch_size
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if _field_kind(f) is FieldKind.CONCAT:
                kwargs[f.name] = _slice_value(val, start, end, bs)
            else:
                kwargs[f.name] = val
        return type(self)(**kwargs)

    def to_device(self: T, device: Union[str, torch.device]) -> T:
        """Move all tensor-like values to *device*."""
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            kwargs[f.name] = _move_value(getattr(self, f.name), device)
        return type(self)(**kwargs)

    def clone(self: T) -> T:
        """Deep-clone the container (tensors are ``.clone()``'d)."""
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            kwargs[f.name] = _clone_value(getattr(self, f.name))
        return type(self)(**kwargs)


__all__ = [
    "Batched",
    "FieldKind",
    "concat_field",
    "shared_field",
    "max_field",
    "min_field",
    "sum_field",
    "mean_field",
]
