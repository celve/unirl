"""Payload helpers for :class:`~diffusionrl.distributed.weight_sync.lora.LoraWeightSync`.

Two pieces turn trainer-side state into what the rollout engine loads:

- ``_peft_config_dict`` — a JSON/Ray-safe PEFT adapter config for the LoRA path
  (``set_lora_from_tensors``).
- ``serialize_named_tensors`` — SGLang ``FlattenedTensorBucket`` payloads (one
  serialized string per dtype) for the merged-full-weight path
  (``update_weights_from_tensor``).

Imported lazily from ``lora.py`` so the driver can reference the handler class
for ``remote(...)`` without eagerly pulling torch / SGLang.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn


def _resolve_peft_config_obj(
    model: nn.Module,
    adapter_name: str = "default",
) -> Any:
    """Walk model wrap layers and return the per-adapter PEFT config object."""
    cur: Any = model
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        pc = getattr(cur, "peft_config", None)
        if isinstance(pc, dict) and adapter_name in pc:
            return pc[adapter_name]
        cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
    return None


def _peft_config_dict(model: nn.Module, adapter_name: str = "default") -> dict:
    """Return a JSON/Ray-safe PEFT config dict for one adapter."""
    peft_cfg_obj = _resolve_peft_config_obj(model, adapter_name)
    if peft_cfg_obj is None:
        raise RuntimeError(f"_peft_config_dict: model has no peft_config[{adapter_name!r}] entry.")

    if hasattr(peft_cfg_obj, "to_dict"):
        peft_dict = peft_cfg_obj.to_dict()
    else:
        peft_dict = dict(peft_cfg_obj)

    tm = peft_dict.get("target_modules")
    if isinstance(tm, str):
        pass
    elif isinstance(tm, (list, tuple, set, frozenset)):
        peft_dict["target_modules"] = sorted(tm) if isinstance(tm, (set, frozenset)) else list(tm)
    elif tm is None:
        pass
    else:
        raise RuntimeError(
            f"_peft_config_dict: peft_config['target_modules'] has "
            f"unsupported type {type(tm).__name__}; expected str / list / "
            f"set / tuple."
        )

    for required in ("r", "lora_alpha", "target_modules"):
        if peft_dict.get(required) in (None, "", [], ()):
            raise RuntimeError(
                f"_peft_config_dict: peft_config[{required!r}] is "
                f"missing or empty (got {peft_dict.get(required)!r}); "
                f"rollout LoRA receive will reject this."
            )

    return peft_dict


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
