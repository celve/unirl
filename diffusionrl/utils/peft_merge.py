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


def extract_lora_tensors(
    model: torch.nn.Module,
    *,
    param_name_prefix: str = "",
    adapter_name: str = "default",
    packed_modules: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Extract LoRA tensors in canonical wire format.

    Canonical format: ``<pipeline_prefix><module>.lora_A.weight`` and
    ``<pipeline_prefix><module>.lora_B.weight`` — PEFT envelope
    (``base_model.model.``) and per-adapter name stripped; pipeline prefix
    retained.  Downstream receivers convert to their engine-specific format:
    :func:`adapt_lora_for_vllm` re-adds the envelope for vllm-omni;
    :func:`adapt_lora_for_sglang` strips the prefix and injects ``.alpha``
    for SGLang.

    ``packed_modules`` maps fused module names to their split sub-names,
    e.g. ``{"qkv_proj": {"sub_names": ["q_proj", "k_proj", "v_proj"], ...}}``.
    When a fused module is found in the LoRA tensors, lora_A is duplicated to
    each sub-name and lora_B is sliced respecting the model's weight layout.
    """
    result: dict[str, torch.Tensor] = {}
    prefix = str(param_name_prefix or "")
    for raw_name, param in model.state_dict().items():
        name = _strip_peft_prefix(raw_name)
        for marker, suffix in ((".lora_A.", "lora_A"), (".lora_B.", "lora_B")):
            if marker not in name:
                continue
            head, adapter_suffix = name.split(marker, 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter != adapter_name:
                break
            out_name = f"{prefix}{head}.{suffix}.weight"
            result[out_name] = _to_full_tensor(param).detach().cpu()
            break

    # Split fused packed-module LoRA tensors into per-sub-layer keys.
    # PEFT wraps a fused module (e.g. qkv_proj) as one LoRA pair:
    #   lora_A: (rank, in_features) — shared across Q/K/V
    #   lora_B: (total_out, rank) — fused, layout model-specific
    # vLLM's MergedQKVParallelLinearWithLoRA.set_lora expects independent
    # (lora_a, lora_b) pairs per sub-layer.
    if packed_modules:
        to_add: dict[str, torch.Tensor] = {}
        to_remove: list[str] = []
        for fused_name, cfg in packed_modules.items():
            # Duck-type: OmegaConf DictConfig is NOT isinstance(cfg, dict).
            # Accept both plain dict and DictConfig-like objects.
            if hasattr(cfg, "get") and "sub_names" in cfg:
                sub_names = list(cfg["sub_names"])
                layout = cfg.get("layout", "block")
                num_q_heads = cfg.get("num_q_heads")
                num_kv_heads = cfg.get("num_kv_heads")
                head_dim = cfg.get("head_dim")
            else:
                sub_names = list(cfg)
                layout = "block"
                num_q_heads = num_kv_heads = head_dim = None

            fused_a = {k: v for k, v in result.items() if f".{fused_name}.lora_A." in k}
            fused_b = {k: v for k, v in result.items() if f".{fused_name}.lora_B." in k}
            for a_key in sorted(fused_a):
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                if b_key not in fused_b:
                    continue
                A = fused_a[a_key]  # (rank, in_features)
                B = fused_b[b_key]  # (total_out, rank)

                if layout == "gqa_interleaved" and num_q_heads and num_kv_heads and head_dim:
                    # HI3-style group-interleaved: B is laid out as
                    # (num_kv_heads, kv_groups+2, head_dim, rank) where
                    # kv_groups = num_q_heads // num_kv_heads.
                    # Split on dim=1: [kv_groups, 1, 1] → Q, K, V
                    kv_groups = num_q_heads // num_kv_heads
                    expected_out = num_kv_heads * (kv_groups + 2) * head_dim
                    rank = B.shape[1]
                    assert B.shape[0] == expected_out, (
                        f"extract_lora_tensors: B.shape[0]={B.shape[0]} != "
                        f"expected {expected_out} for gqa_interleaved "
                        f"(num_q={num_q_heads}, num_kv={num_kv_heads}, "
                        f"head_dim={head_dim})"
                    )
                    B_4d = B.reshape(num_kv_heads, kv_groups + 2, head_dim, rank)
                    B_q, B_k, B_v = torch.split(B_4d, [kv_groups, 1, 1], dim=1)
                    split_tensors = [
                        B_q.reshape(num_q_heads * head_dim, rank).contiguous(),
                        B_k.reshape(num_kv_heads * head_dim, rank).contiguous(),
                        B_v.reshape(num_kv_heads * head_dim, rank).contiguous(),
                    ]
                else:
                    # Block layout: simple contiguous split
                    n = len(sub_names)
                    assert B.shape[0] % n == 0, (
                        f"extract_lora_tensors: B.shape[0]={B.shape[0]} not "
                        f"divisible by {n} for block split of {fused_name}."
                    )
                    step = B.shape[0] // n
                    split_tensors = [B[i * step : (i + 1) * step, :].contiguous() for i in range(n)]

                base_prefix = a_key.replace(f".{fused_name}.lora_A.weight", "")
                for i, sub in enumerate(sub_names):
                    to_add[f"{base_prefix}.{sub}.lora_A.weight"] = A.clone()
                    to_add[f"{base_prefix}.{sub}.lora_B.weight"] = split_tensors[i]
                to_remove.append(a_key)
                to_remove.append(b_key)
        for k in to_remove:
            result.pop(k, None)
        result.update(to_add)

    # Defensive dtype check: vllm punica kernel hard-asserts inputs.dtype in
    # {fp16, bf16}. Catch fp32 LoRA here in trainer (cheap) rather than
    # crashing ~20min later in rollout.
    _bad_dtype = [
        (k, v.dtype) for k, v in result.items() if ".lora_" in k and v.dtype not in (torch.bfloat16, torch.float16)
    ]
    if _bad_dtype:
        sample = ", ".join(f"{k}={dt}" for k, dt in _bad_dtype[:3])
        raise RuntimeError(
            f"extract_lora_tensors: {len(_bad_dtype)} LoRA tensor(s) have "
            f"unsupported dtype for vllm punica kernel (expected bf16/fp16). "
            f"Sample: [{sample}]. Check FSDP MixedPrecisionPolicy.param_dtype."
        )

    return result


def adapt_lora_for_vllm(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Wrap canonical-format LoRA keys in the vllm-omni PEFT envelope.

    Canonical → vllm-omni format::

        <pipeline_prefix><module>.lora_A.weight
        → base_model.model.<pipeline_prefix><module>.lora_A.weight

    This is the receiver-side adapter for
    :class:`~diffusionrl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine`.
    """
    return {f"{_PEFT_PREFIX}{k}": v for k, v in tensors.items()}


def adapt_lora_for_sglang(
    tensors: dict[str, torch.Tensor],
    *,
    pipeline_prefix: str = "",
    peft_config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Convert canonical-format LoRA tensors to SGLang's native key format.

    Canonical → SGLang native::

        <pipeline_prefix><module>.lora_A.weight
        → <module>.lora_A.weight
        + <module>.alpha            ← injected from peft_config["lora_alpha"]

    SGLang's ``_apply_lora_to_layers`` keys its ``lora_layers`` dict by
    ``named_modules()`` of ``self.modules["transformer"]`` — i.e. starting
    *inside* the transformer — so layer keys are bare module names without
    any pipeline prefix.  The ``.alpha`` key is required so SGLang computes
    ``scale = lora_alpha / r`` correctly; without it SGLang falls back to
    ``inferred_alpha = inferred_rank`` → scale = 1.0 (wrong for alpha ≠ rank).

    Args:
        tensors: Canonical-format output of :func:`extract_lora_tensors`.
        pipeline_prefix: The pipeline-level prefix to strip, e.g.
            ``"transformer."`` for SD3/WAN/HV15/Qwen or ``"model."`` for
            HunyuanImage3.  Read from
            ``model_config.weight_sync_param_name_prefix`` at the call site.
        peft_config: PEFT config dict; provides ``lora_alpha`` for injecting
            ``.alpha`` keys.
    """
    prefix = str(pipeline_prefix or "")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if prefix and key.startswith(prefix):
            key = key[len(prefix):]
        result[key] = tensor

    if peft_config:
        lora_alpha = peft_config.get("lora_alpha")
        if lora_alpha is not None:
            layer_bases: set[str] = set()
            for k in result:
                for suf in (".lora_A.weight", ".lora_A"):
                    if k.endswith(suf):
                        layer_bases.add(k[: -len(suf)])
                        break
            alpha_tensor = torch.tensor(float(lora_alpha))
            for base in layer_bases:
                alpha_key = f"{base}.alpha"
                if alpha_key not in result:
                    result[alpha_key] = alpha_tensor

    return result
