"""Shared batch-shaped payload helpers for rollout and training types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

if TYPE_CHECKING:
    from torch import device as TorchDevice


def copy_columnar_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(mapping, Mapping):
        return result
    for key, value in mapping.items():
        if isinstance(value, list):
            result[str(key)] = list(value)
        elif isinstance(value, tuple):
            result[str(key)] = list(value)
        else:
            result[str(key)] = value
    return result


def slice_columnar_value(
    value: Any,
    *,
    batch_size: int,
    start: int,
    end: int,
) -> Any:
    if value is None:
        return None
    if isinstance(value, list) and len(value) == batch_size:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[start:end]
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[start:end].clone()
    if hasattr(value, "slice") and callable(getattr(value, "slice")):
        return value.slice(start, end)
    return value


def pad_columnar_value(
    value: Any,
    *,
    batch_size: int,
    target_size: int,
) -> Any:
    if value is None or target_size <= batch_size:
        return value

    pad_count = target_size - batch_size
    if isinstance(value, list) and len(value) == batch_size:
        return value + [value[-1]] * pad_count
    if isinstance(value, tuple) and len(value) == batch_size:
        return value + (value[-1],) * pad_count
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        pad = value[-1:].repeat(pad_count, *([1] * (value.dim() - 1)))
        return torch.cat([value, pad], dim=0)
    return value


def concat_columnar_values(
    values: List[Any],
    *,
    batch_sizes: List[int],
) -> Any:
    non_none = [value for value in values if value is not None]
    if not non_none:
        return None

    if all(
        isinstance(value, list) and len(value) == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        merged: List[Any] = []
        for value in values:
            if value is not None:
                merged.extend(list(value))
        return merged
    if all(
        isinstance(value, tuple) and len(value) == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        merged_list: List[Any] = []
        for value in values:
            if value is not None:
                merged_list.extend(list(value))
        return tuple(merged_list)
    if all(
        isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        return torch.cat([value for value in values if value is not None], dim=0)
    if all(isinstance(value, Mapping) for value in non_none):
        merged_mapping: Dict[str, Any] = {}
        keys = sorted({str(key) for value in non_none for key in value.keys()})
        for key in keys:
            merged_mapping[key] = concat_columnar_values(
                [
                    dict(value).get(key) if isinstance(value, Mapping) else None
                    for value in values
                ],
                batch_sizes=batch_sizes,
            )
        return merged_mapping

    first = non_none[0]
    if torch.is_tensor(first):
        if all(torch.is_tensor(value) and torch.equal(value, first) for value in non_none[1:]):
            return first
    elif all(value == first for value in non_none[1:]):
        return first
    raise ValueError(
        "Cannot concatenate rollout values with mismatched non-batched content: "
        f"types={[type(value).__name__ if value is not None else None for value in values]}"
    )


def clone_payload_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {str(k): clone_payload_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_payload_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_payload_value(v) for v in value)
    if isinstance(value, set):
        return {clone_payload_value(v) for v in value}
    return value


def move_payload_value(value: Any, device: Union[str, "TorchDevice"]) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {str(k): move_payload_value(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_payload_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_payload_value(v, device) for v in value)
    if isinstance(value, set):
        return {move_payload_value(v, device) for v in value}
    return value


def slice_payload_value(value: Any, *, start: int, end: int, batch_size: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(k): slice_payload_value(v, start=start, end=end, batch_size=batch_size)
            for k, v in value.items()
        }
    if isinstance(value, list) and len(value) == batch_size:
        return [clone_payload_value(v) for v in value[start:end]]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(clone_payload_value(v) for v in value[start:end])
    if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == batch_size:
        return value[start:end].clone()
    return clone_payload_value(value)


def reindex_payload_value(value: Any, *, indices: torch.Tensor, batch_size: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(k): reindex_payload_value(v, indices=indices, batch_size=batch_size)
            for k, v in value.items()
        }
    index_list = indices.tolist()
    if isinstance(value, list) and len(value) == batch_size:
        return [clone_payload_value(value[i]) for i in index_list]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(clone_payload_value(value[i]) for i in index_list)
    if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == batch_size:
        return value[indices.to(value.device)]
    return clone_payload_value(value)


def concat_payload_values(values: List[Any], *, batch_sizes: List[int]) -> Any:
    non_none = [value for value in values if value is not None]
    if not non_none:
        return None

    if all(isinstance(value, dict) for value in non_none):
        keys = sorted({str(k) for value in non_none for k in value.keys()})
        return {
            key: concat_payload_values(
                [value.get(key) if isinstance(value, dict) else None for value in values],
                batch_sizes=batch_sizes,
            )
            for key in keys
        }

    if all(
        isinstance(value, list) and len(value) == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        merged: List[Any] = []
        for value in values:
            if value is None:
                continue
            merged.extend(clone_payload_value(item) for item in value)
        return merged

    if all(
        isinstance(value, tuple) and len(value) == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        merged_tuple: List[Any] = []
        for value in values:
            if value is None:
                continue
            merged_tuple.extend(clone_payload_value(item) for item in value)
        return tuple(merged_tuple)

    if all(
        torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == batch_size
        for value, batch_size in zip(values, batch_sizes)
        if value is not None
    ):
        return torch.cat([value for value in values if value is not None], dim=0)

    first = non_none[0]
    if all(value == first for value in non_none[1:]):
        return clone_payload_value(first)
    return [clone_payload_value(value) for value in values]


__all__ = [
    "clone_payload_value",
    "concat_columnar_values",
    "concat_payload_values",
    "copy_columnar_mapping",
    "move_payload_value",
    "pad_columnar_value",
    "reindex_payload_value",
    "slice_columnar_value",
    "slice_payload_value",
]
