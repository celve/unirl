"""Shared SGLang-shape tensor serialization for weight sync.

Both the v1 ``UpdateWeightFromTensor`` handler and the v2 sibling
``LoraWeightSync`` (merged mode) pack named tensors into SGLang
``FlattenedTensorBucket`` payloads — one serialized string per dtype — that the
rollout engine reconstructs with ``load_format="flattened_bucket"``. This helper
is the single source of that packing so both senders stay byte-identical.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def serialize_named_tensors(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> List[str]:
    """Pack ``(name, tensor)`` pairs into one serialized payload per dtype.

    Groups by dtype (insertion order preserved), builds one
    ``FlattenedTensorBucket`` per group, and serializes each via
    ``MultiprocessingSerializer``. SGLang imports are deferred so a driver /
    non-rollout process can import this module without pulling SGLang.
    """
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

    named_tensors_by_dtype: dict = {}
    for name, tensor in named_tensors:
        named_tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

    serialized: List[str] = []
    for grouped_named_tensors in named_tensors_by_dtype.values():
        bucket = FlattenedTensorBucket(named_tensors=grouped_named_tensors)
        payload = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized.append(MultiprocessingSerializer.serialize(payload, output_str=True))
    return serialized


__all__ = ["serialize_named_tensors"]
