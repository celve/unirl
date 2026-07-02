"""Standalone mcore Qwen3 probe — de-risk model build + HF<->mcore converter + forward
correctness BEFORE integrating into the unirl backend. Validates against HF transformers
as ground truth. TE-free (local spec), single GPU, padded forward.

Run on the pod:
  cd /root/unirl && source .venv-sglang/bin/activate
  MODEL=/root/unirl/models/local/Qwen3-0.6B python scripts/megatron_probe/probe.py
"""
import os, sys, json, glob
import torch
import torch.nn.functional as F

MODEL = os.environ.get("MODEL", "/root/unirl/models/local/Qwen3-0.6B")


def log(*a):
    print(*a, flush=True)


def init_dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)
    log("[stage0] dist + parallel_state initialized")


def build_config(hf):
    from megatron.core.transformer.transformer_config import TransformerConfig
    cfg = TransformerConfig(
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
    return cfg


def build_model(cfg, hf):
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
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


def load_hf_state(path):
    st = {}
    from safetensors.torch import load_file
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    for s in shards:
        st.update(load_file(s))
    return st


def hf_to_mcore(model, hf_sd, hf):
    """Fill mcore params from HF state dict. Local spec names (verified stage1)."""
    n_group = hf["num_key_value_heads"]
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    hidden = hf["hidden_size"]
    vpg = hf["num_attention_heads"] // n_group  # value_num_per_group
    md = dict(model.named_parameters())
    used_hf, filled = set(), set()

    def put(mname, tensor):
        p = md.get(mname)
        if p is None:
            return False
        assert p.shape == tensor.shape, f"{mname}: mcore {tuple(p.shape)} != conv {tuple(tensor.shape)}"
        with torch.no_grad():
            p.copy_(tensor.to(p.dtype).to(p.device))
        filled.add(mname)
        return True

    def g(k):
        used_hf.add(k)
        return hf_sd[k].to(torch.bfloat16)

    put("embedding.word_embeddings.weight", g("model.embed_tokens.weight"))
    put("decoder.final_layernorm.weight", g("model.norm.weight"))
    if not hf.get("tie_word_embeddings", False):
        put("output_layer.weight", g("lm_head.weight"))
    else:
        used_hf.add("lm_head.weight")  # tied; no separate param

    for i in range(hf["num_hidden_layers"]):
        H = f"model.layers.{i}."
        M = f"decoder.layers.{i}."
        # fused QKV: HF q/k/v -> mcore linear_qkv, interleaved per group [q..,k,v]
        q = g(H + "self_attn.q_proj.weight").view(n_group, vpg, head_dim, hidden)
        k = g(H + "self_attn.k_proj.weight").view(n_group, 1, head_dim, hidden)
        v = g(H + "self_attn.v_proj.weight").view(n_group, 1, head_dim, hidden)
        qkv = torch.cat([q, k, v], dim=1).reshape(-1, hidden)
        put(M + "self_attention.linear_qkv.weight", qkv)
        put(M + "self_attention.linear_proj.weight", g(H + "self_attn.o_proj.weight"))
        put(M + "self_attention.q_layernorm.weight", g(H + "self_attn.q_norm.weight"))
        put(M + "self_attention.k_layernorm.weight", g(H + "self_attn.k_norm.weight"))
        # gate_up fused
        gate = g(H + "mlp.gate_proj.weight")
        up = g(H + "mlp.up_proj.weight")
        put(M + "mlp.linear_fc1.weight", torch.cat([gate, up], dim=0))
        put(M + "mlp.linear_fc2.weight", g(H + "mlp.down_proj.weight"))
        # layernorms — try BOTH fused (TE-style) and separate (local) names
        iln = g(H + "input_layernorm.weight")
        if not put(M + "input_layernorm.weight", iln):
            put(M + "self_attention.linear_qkv.layer_norm_weight", iln)
        pln = g(H + "post_attention_layernorm.weight")
        if not put(M + "pre_mlp_layernorm.weight", pln):
            put(M + "mlp.linear_fc1.layer_norm_weight", pln)

    missing = [n for n in md if n not in filled]
    unused = [k for k in hf_sd if k not in used_hf]
    return filled, missing, unused


def mcore_to_hf(name, param, hf):
    """mcore param -> list of (hf_name, tensor). Inverse of hf_to_mcore (local spec)."""
    import re
    n_group = hf["num_key_value_heads"]
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    hidden = hf["hidden_size"]
    vpg = hf["num_attention_heads"] // n_group
    if name == "embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]
    if name == "output_layer.weight":
        return [] if hf.get("tie_word_embeddings", False) else [("lm_head.weight", param)]
    m = re.match(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not m:
        return []
    i, rest = m.groups()
    L = f"model.layers.{i}."
    if rest == "self_attention.linear_qkv.weight":
        p = param.view(n_group, vpg + 2, head_dim, hidden)
        q, k, v = torch.split(p, [vpg, 1, 1], dim=1)
        return [(L + "self_attn.q_proj.weight", q.reshape(-1, hidden)),
                (L + "self_attn.k_proj.weight", k.reshape(-1, hidden)),
                (L + "self_attn.v_proj.weight", v.reshape(-1, hidden))]
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
    raise ValueError(f"mcore_to_hf: unmapped {name}")


def main():
    hf = json.load(open(os.path.join(MODEL, "config.json")))
    log("[cfg]", {k: hf[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                     "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings")})
    init_dist()

    cfg = build_config(hf)
    model = build_model(cfg, hf)
    log("[stage1] GPTModel built. named_parameters (first 20):")
    for i, (n, p) in enumerate(model.named_parameters()):
        if i < 20 or "layers.0." in n:
            log("   ", n, tuple(p.shape))
    log("   ... total params:", sum(1 for _ in model.named_parameters()))

    log("[stage2] loading HF weights via converter...")
    hf_sd = load_hf_state(MODEL)
    filled, missing, unused = hf_to_mcore(model, hf_sd, hf)
    log(f"   filled={len(filled)} missing={len(missing)} unused_hf={len(unused)}")
    if missing:
        log("   MISSING mcore params:", missing[:12])
    if unused:
        log("   UNUSED hf keys:", unused[:12])
    if missing:
        log("[stage2] FAILED — fix converter names above. Stopping.")
        return

    log("[stage3] forward vs HF transformers ground truth...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    S = ids.shape[1]
    with torch.no_grad():
        hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
        hf_logits = hf_model(ids).logits  # [1,S,V]
        # mcore padded causal forward
        pos = torch.arange(S, device=ids.device).unsqueeze(0)
        amask = torch.triu(torch.ones(1, 1, S, S, device=ids.device, dtype=torch.bool), diagonal=1)
        out = model(input_ids=ids, position_ids=pos, attention_mask=amask)
        mc_logits = out.transpose(0, 1) if out.shape[0] == S else out  # -> [1,S,V]
    hf_arg = hf_logits[0].argmax(-1)
    mc_arg = mc_logits[0, :, : hf_logits.shape[-1]].argmax(-1)
    agree = (hf_arg == mc_arg).float().mean().item()
    log(f"   argmax agreement mcore-vs-HF = {agree:.3f} (want ~1.0)")
    if agree <= 0.9:
        log("[stage3] FORWARD MISMATCH — stopping.")
        return

    log("[stage4] mcore->HF export round-trip check...")
    exported = {}
    for n, p in model.named_parameters():
        for hn, ht in mcore_to_hf(n, p.detach(), hf):
            exported[hn] = ht
    max_err, nbad = 0.0, 0
    for k, orig in hf_sd.items():
        if k == "lm_head.weight" and hf.get("tie_word_embeddings", False):
            continue
        if k not in exported:
            log("   EXPORT MISSING:", k); nbad += 1; continue
        e = (exported[k].float() - orig.float().cuda()).abs().max().item()
        max_err = max(max_err, e)
    log(f"   export keys={len(exported)} max_abs_err={max_err:.2e} missing={nbad} (want err=0, missing=0)")

    log("[stage5] DDP wrap + dist-optimizer + one train step...")
    from megatron.core.distributed import DistributedDataParallel as DDP, DistributedDataParallelConfig
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig as McoreOptCfg
    from megatron.core.pipeline_parallel import get_forward_backward_func
    ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=True, overlap_grad_reduce=False)
    ddp = DDP(cfg, ddp_cfg, model)
    opt_cfg = McoreOptCfg(optimizer="adam", lr=1e-5, weight_decay=0.0, bf16=True,
                          use_distributed_optimizer=True, params_dtype=torch.bfloat16)
    optimizer = get_megatron_optimizer(opt_cfg, [ddp])
    log("   DDP + optimizer built")

    def forward_step(data_iter, mdl):
        batch = next(data_iter)
        out = mdl(input_ids=batch["ids"], position_ids=batch["pos"], attention_mask=batch["mask"])

        def loss_fn(logits):
            lg = logits.transpose(0, 1) if logits.shape[0] == batch["ids"].shape[1] else logits
            lp = torch.log_softmax(lg[:, :-1].float(), dim=-1)
            tgt = batch["ids"][:, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
            return nll * 1.0, torch.tensor(1, device=logits.device), {"keys": [], "values": logits.new_zeros(1)}
        return out, loss_fn

    for chunk in [ddp]:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    data = iter([{"ids": ids, "pos": pos, "mask": amask}])
    losses = get_forward_backward_func()(
        forward_step_func=forward_step, data_iterator=data, model=[ddp],
        num_microbatches=1, seq_length=S, micro_batch_size=1, forward_only=False)
    ok, gnorm, nz = optimizer.step()
    log(f"   train step: success={ok} grad_norm={float(gnorm):.4f} losses={losses}")
    log("[done] ALL STAGES PASSED." if (max_err < 1e-2 and nbad == 0 and ok) else "[done] see failures above.")


if __name__ == "__main__":
    main()
