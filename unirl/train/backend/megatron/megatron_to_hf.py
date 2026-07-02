"""mcore -> HF weight-name/shape converter (Qwen2/Qwen3 dense, local spec).

Inverse of ``model_provider.load_hf_weights``; validated round-trip err 0.0 against
the original HF safetensors (scripts/megatron_probe/probe.py stage4). Ported from
slime ``megatron_to_hf/qwen2.py`` but for the LOCAL layer spec (separate
``input_layernorm`` / ``pre_mlp_layernorm`` modules, not the TE-fused
``linear_qkv.layer_norm_weight``).

1 mcore param -> N HF (name, tensor): qkv de-fusion by query group, gate/up chunk,
q/k-norm + layernorm renames. Tied embeddings emit no separate ``lm_head`` (SGLang
aliases it to ``embed_tokens``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch


def convert_mcore_to_hf(name: str, param: torch.Tensor, hf: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    n_group = hf["num_key_value_heads"]
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    hidden = hf["hidden_size"]
    vpg = hf["num_attention_heads"] // n_group  # value_num_per_group

    if name == "embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]
    if name == "output_layer.weight":
        # Tied: no separate lm_head (SGLang aliases embed_tokens). Untied: emit it.
        return [] if hf.get("tie_word_embeddings", False) else [("lm_head.weight", param)]

    m = re.match(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not m:
        raise ValueError(f"convert_mcore_to_hf: unmapped param {name!r}")
    i, rest = m.groups()
    L = f"model.layers.{i}."

    if rest == "self_attention.linear_qkv.weight":
        p = param.view(n_group, vpg + 2, head_dim, hidden)
        q, k, v = torch.split(p, [vpg, 1, 1], dim=1)
        return [
            (L + "self_attn.q_proj.weight", q.reshape(-1, hidden)),
            (L + "self_attn.k_proj.weight", k.reshape(-1, hidden)),
            (L + "self_attn.v_proj.weight", v.reshape(-1, hidden)),
        ]
    if rest == "self_attention.linear_proj.weight":
        return [(L + "self_attn.o_proj.weight", param)]
    if rest == "self_attention.q_layernorm.weight":
        return [(L + "self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(L + "self_attn.k_norm.weight", param)]
    if rest == "input_layernorm.weight":
        return [(L + "input_layernorm.weight", param)]
    if rest == "pre_mlp_layernorm.weight":
        return [(L + "post_attention_layernorm.weight", param)]
    if rest == "mlp.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [(L + "mlp.gate_proj.weight", gate), (L + "mlp.up_proj.weight", up)]
    if rest == "mlp.linear_fc2.weight":
        return [(L + "mlp.down_proj.weight", param)]
    raise ValueError(f"convert_mcore_to_hf: unmapped param {name!r}")
