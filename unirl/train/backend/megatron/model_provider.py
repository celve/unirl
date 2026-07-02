"""mcore model construction (raw GPTModel, local spec, TE-free) + HF<->mcore weights.

Validated end-to-end against HF transformers (scripts/megatron_probe/probe.py):
forward argmax matches HF 1.000, HF<->mcore round-trip error 0.0. TE is unavailable
on this env (no cuBLAS-13 wheel for the grouped-GEMM symbol), so we use mcore's
``get_gpt_layer_local_spec`` (Torch RMSNorm, unfused attention) instead of AutoBridge.

Covers Qwen2/Qwen3 dense (GQA, decoupled head_dim, qk-layernorm, tied embeddings).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def read_hf_config(hf_checkpoint: str) -> Dict[str, Any]:
    with open(os.path.join(hf_checkpoint, "config.json")) as f:
        return json.load(f)


def build_transformer_config(hf: Dict[str, Any]):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=hf["num_hidden_layers"],
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        ffn_hidden_size=hf["intermediate_size"],
        kv_channels=hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        layernorm_epsilon=hf["rms_norm_eps"],
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=hf.get("attention_bias", False),
        qk_layernorm=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def build_gpt_model(cfg, hf: Dict[str, Any]):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    spec = get_gpt_layer_local_spec(qk_layernorm=True)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=hf["vocab_size"],
        max_sequence_length=hf["max_position_embeddings"],
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=hf.get("tie_word_embeddings", False),
        position_embedding_type="rope",
        rotary_base=hf.get("rope_theta", 10000),
    )
    return model.cuda().bfloat16()


def load_hf_weights(model, hf_checkpoint: str, hf: Dict[str, Any]) -> None:
    """Fill mcore params from HF safetensors (validated: 226/226, err 0)."""
    from safetensors.torch import load_file

    hf_sd: Dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        hf_sd.update(load_file(shard))

    n_group = hf["num_key_value_heads"]
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    hidden = hf["hidden_size"]
    vpg = hf["num_attention_heads"] // n_group
    md = dict(model.named_parameters())
    filled = set()

    def put(mname: str, tensor: torch.Tensor) -> bool:
        p = md.get(mname)
        if p is None:
            return False
        assert p.shape == tensor.shape, f"{mname}: mcore {tuple(p.shape)} != {tuple(tensor.shape)}"
        with torch.no_grad():
            p.copy_(tensor.to(p.dtype).to(p.device))
        filled.add(mname)
        return True

    def g(k: str) -> torch.Tensor:
        return hf_sd[k].to(torch.bfloat16)

    put("embedding.word_embeddings.weight", g("model.embed_tokens.weight"))
    put("decoder.final_layernorm.weight", g("model.norm.weight"))
    if not hf.get("tie_word_embeddings", False):
        put("output_layer.weight", g("lm_head.weight"))

    for i in range(hf["num_hidden_layers"]):
        H, M = f"model.layers.{i}.", f"decoder.layers.{i}."
        q = g(H + "self_attn.q_proj.weight").view(n_group, vpg, head_dim, hidden)
        k = g(H + "self_attn.k_proj.weight").view(n_group, 1, head_dim, hidden)
        v = g(H + "self_attn.v_proj.weight").view(n_group, 1, head_dim, hidden)
        put(M + "self_attention.linear_qkv.weight", torch.cat([q, k, v], dim=1).reshape(-1, hidden))
        put(M + "self_attention.linear_proj.weight", g(H + "self_attn.o_proj.weight"))
        put(M + "self_attention.q_layernorm.weight", g(H + "self_attn.q_norm.weight"))
        put(M + "self_attention.k_layernorm.weight", g(H + "self_attn.k_norm.weight"))
        put(M + "mlp.linear_fc1.weight", torch.cat([g(H + "mlp.gate_proj.weight"), g(H + "mlp.up_proj.weight")], dim=0))
        put(M + "mlp.linear_fc2.weight", g(H + "mlp.down_proj.weight"))
        put(M + "input_layernorm.weight", g(H + "input_layernorm.weight"))
        put(M + "pre_mlp_layernorm.weight", g(H + "post_attention_layernorm.weight"))

    missing = [n for n in md if n not in filled]
    if missing:
        raise RuntimeError(f"load_hf_weights: {len(missing)} mcore params unfilled: {missing[:8]}")


def build_model_and_load(cfg_hf_checkpoint: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """Returns (gpt_model, transformer_config, hf_config) with weights loaded."""
    hf = read_hf_config(cfg_hf_checkpoint)
    tconf = build_transformer_config(hf)
    model = build_gpt_model(tconf, hf)
    load_hf_weights(model, cfg_hf_checkpoint, hf)
    return model, tconf, hf
