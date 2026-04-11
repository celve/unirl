"""Utilities for merging or extracting PEFT/LoRA weights."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.distributed.tensor import DTensor, Replicate

_PEFT_PREFIX = "base_model.model."


def _strip_peft_prefix(name: str) -> str:
    return name.removeprefix(_PEFT_PREFIX)


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize DTensor parameters into regular tensors on CUDA."""
    tensor = tensor.cuda()
    if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim).to_local()
    return tensor


def merged_state_dict(
    model: torch.nn.Module,
    adapter_name: str = "default",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield `(name, tensor)` pairs with LoRA deltas folded into base weights."""
    if not hasattr(model, "peft_config"):
        for name, param in model.state_dict().items():
            yield (name, _to_full_tensor(param))
        return

    peft_cfg = model.peft_config[adapter_name]
    scaling = peft_cfg.lora_alpha / peft_cfg.r
    state_dict = model.state_dict()

    lora_groups: dict[str, dict[str, str]] = {}
    regular_keys: list[str] = []

    for raw_name in state_dict:
        name = _strip_peft_prefix(raw_name)

        if ".base_layer." in name:
            original = name.replace(".base_layer.", ".")
            lora_groups.setdefault(original, {})["base"] = raw_name
        elif ".lora_A." in name:
            prefix, adapter_suffix = name.split(".lora_A.", 1)
            adapter, *rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                original = prefix + "." + rest[0] if rest else prefix
                lora_groups.setdefault(original, {})["lora_A"] = raw_name
        elif ".lora_B." in name:
            prefix, adapter_suffix = name.split(".lora_B.", 1)
            adapter, *rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                original = prefix + "." + rest[0] if rest else prefix
                lora_groups.setdefault(original, {})["lora_B"] = raw_name
        else:
            regular_keys.append(raw_name)

    for original_name, group in lora_groups.items():
        if "base" not in group:
            continue
        base = _to_full_tensor(state_dict[group["base"]])
        if "lora_A" in group and "lora_B" in group:
            lora_a = _to_full_tensor(state_dict[group["lora_A"]])
            lora_b = _to_full_tensor(state_dict[group["lora_B"]])
            yield (original_name, base + (lora_b @ lora_a) * scaling)
        else:
            yield (original_name, base)

    for raw_name in regular_keys:
        yield (_strip_peft_prefix(raw_name), _to_full_tensor(state_dict[raw_name]))


def raw_state_dict(
    model: torch.nn.Module,
    adapter_name: str = "default",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield base and LoRA weights separately, matching rollout-engine naming."""
    if not hasattr(model, "peft_config"):
        for name, param in model.state_dict().items():
            yield (name, _to_full_tensor(param))
        return

    state_dict = model.state_dict()

    base_names: dict[str, str] = {}
    lora_a_keys: dict[str, str] = {}
    lora_b_keys: dict[str, str] = {}
    regular_keys: list[str] = []

    for raw_name in state_dict:
        name = _strip_peft_prefix(raw_name)

        if ".base_layer." in name:
            base_names[raw_name] = name
        elif ".lora_A." in name:
            prefix, adapter_suffix = name.split(".lora_A.", 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                lora_a_keys[prefix] = raw_name
        elif ".lora_B." in name:
            prefix, adapter_suffix = name.split(".lora_B.", 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                lora_b_keys[prefix] = raw_name
        else:
            regular_keys.append(raw_name)

    for raw_name, stripped_name in base_names.items():
        yield (stripped_name.replace(".base_layer.", "."), _to_full_tensor(state_dict[raw_name]))
    for prefix, raw_name in lora_a_keys.items():
        yield (prefix + ".lora_A", _to_full_tensor(state_dict[raw_name]))
    for prefix, raw_name in lora_b_keys.items():
        yield (prefix + ".lora_B", _to_full_tensor(state_dict[raw_name]))
    for raw_name in regular_keys:
        yield (_strip_peft_prefix(raw_name), _to_full_tensor(state_dict[raw_name]))
