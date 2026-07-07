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


def build_transformer_config(hf: Dict[str, Any], tp_size: int = 1, pp_size: int = 1):
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
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        # SP off: each TP rank computes the full (redundant) layernorm — correct,
        # just less memory-efficient. SP is a later optimization.
        sequence_parallel=False,
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
        # parallel_output=False -> the output layer all-gathers vocab-parallel logits
        # to full [b,s,V] on every TP rank, so the loss bridge needs no vocab-parallel
        # reduction (correctness-first; vocab-parallel logp is a later memory win).
        parallel_output=False,
    )
    return model.cuda().bfloat16()


def _tp_shard(p: torch.Tensor, full: torch.Tensor, *, tp_rank: int, tp_size: int, glu: bool) -> torch.Tensor:
    """Slice a full HF-equivalent tensor into this TP rank's shard.

    Non-TP params (layernorms, etc.) are replicated (returned whole). TP params
    slice along the mcore param's ``partition_dim``. GLU ``linear_fc1`` is special:
    mcore stores it as ``[gate; up]`` but shards each half so every rank holds
    ``[gate_r; up_r]`` — a naive dim-0 slice would give one rank all-gate.
    """
    if tp_size == 1 or not getattr(p, "tensor_model_parallel", False):
        return full
    if glu:
        gate, up = full.chunk(2, dim=0)
        return torch.cat([gate.chunk(tp_size, 0)[tp_rank], up.chunk(tp_size, 0)[tp_rank]], dim=0).contiguous()
    dim = int(getattr(p, "partition_dim", 0))
    # mcore may pad the vocab (dim 0) to a multiple of make_vocab_size_divisible_by;
    # pad the full tensor up to the mcore full size before slicing.
    mcore_full = p.shape[dim] * tp_size
    if full.shape[dim] < mcore_full:
        pad = [0, 0] * full.dim()
        pad[-(2 * dim + 1)] = mcore_full - full.shape[dim]  # pad the end of `dim`
        full = torch.nn.functional.pad(full, pad)
    return full.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def load_hf_weights(model, hf_checkpoint: str, hf: Dict[str, Any], *, tp_rank: int = 0, tp_size: int = 1) -> None:
    """Fill mcore params from HF safetensors, TP-sharded per rank.

    Validated at tp=1 (226/226, err 0). Under tp>1 each mcore param is this rank's
    shard: build the full fused HF-equivalent tensor (same as tp=1), then slice.
    """
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

    def put(mname: str, full: torch.Tensor, glu: bool = False) -> bool:
        p = md.get(mname)
        if p is None:
            return False
        shard = _tp_shard(p, full, tp_rank=tp_rank, tp_size=tp_size, glu=glu)
        assert p.shape == shard.shape, f"{mname}: mcore {tuple(p.shape)} != {tuple(shard.shape)}"
        with torch.no_grad():
            p.copy_(shard.to(p.dtype).to(p.device))
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
        put(M + "mlp.linear_fc1.weight", torch.cat([g(H + "mlp.gate_proj.weight"), g(H + "mlp.up_proj.weight")], dim=0), glu=True)
        put(M + "mlp.linear_fc2.weight", g(H + "mlp.down_proj.weight"))
        put(M + "input_layernorm.weight", g(H + "input_layernorm.weight"))
        put(M + "pre_mlp_layernorm.weight", g(H + "post_attention_layernorm.weight"))

    missing = [n for n in md if n not in filled]
    if missing:
        raise RuntimeError(f"load_hf_weights: {len(missing)} mcore params unfilled: {missing[:8]}")


def build_model_and_load(hf_checkpoint: str, *, tp_size: int = 1, pp_size: int = 1,
                         tp_rank: int = 0) -> Tuple[Any, Any, Dict[str, Any]]:
    """Returns (gpt_model, transformer_config, hf_config) with TP-sharded weights loaded."""
    hf = read_hf_config(hf_checkpoint)
    tconf = build_transformer_config(hf, tp_size=tp_size, pp_size=pp_size)
    model = build_gpt_model(tconf, hf)
    load_hf_weights(model, hf_checkpoint, hf, tp_rank=tp_rank, tp_size=tp_size)
    return model, tconf, hf
