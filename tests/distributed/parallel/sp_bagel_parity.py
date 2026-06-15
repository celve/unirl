"""BAGEL packed-MoT Ulysses SP parity: sp=1 reference vs sp=N (forward + backward).

Validates the BAGEL SP adapter (unirl.train.backend.veomni.sp.bagel) — the query-
stream slice/gather boundary + the flash_attn_varlen_func Ulysses wrap — on a small
real vendored Qwen2ForCausalLM(MoT), driven through ``forward_inference(mode="gen")``
with a synthetic replicated text KV cache (cross-length q/k, causal=False), exactly
the geometry of ``Bagel._forward_flow``.

Run sp=1 first (saves weights + inputs + reference outputs/grads), then sp=N:

  torchrun --nproc_per_node=1 --master_port=29590 /tmp/sp_bagel_parity.py   # ref
  torchrun --nproc_per_node=4 --master_port=29591 /tmp/sp_bagel_parity.py   # compare

Use sp=4 (NOT just sp=2): with L_q=14 and sp=4 the query stream pads (14→16), so sp=4
exercises the pad-strip path that sp=2 (no pad) would silently skip; ranks 1-2 hold an
all-VAE slice (empty text index); rank 3 holds a pad tail. GQA (8q/4kv) is exercised.

Forward parity: gathered output vs the sp=1 reference. Backward parity: the model
params are replicated (no FSDP here), so each SP rank holds its query slice's grad;
summing across the SP group reconstructs the full grad (the folded-mesh property that
sp_fsdp.py validates with real FSDP). bf16 reduction-order residual ~1e-2; a real SP
bug is a much larger, systematic diff.
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism

ST = "/tmp/bagel_sp_state.pt"   # weights + inputs + cache (shared across launches)
REF = "/tmp/bagel_sp_ref.pt"    # sp=1 reference output + grad

N_LATENT = 12                   # query = [SOI, 12 latents, EOI] -> L_q = 14
L_CTX = 8                       # replicated text-context cache length
HEADS_Q, HEADS_KV, HEAD_DIM = 8, 4, 16
HIDDEN = HEADS_Q * HEAD_DIM     # 128
N_LAYERS = 2


def build_model(dev):
    from unirl.models.bagel.vendor.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=256, num_hidden_layers=N_LAYERS,
        num_attention_heads=HEADS_Q, num_key_value_heads=HEADS_KV, head_dim=HEAD_DIM,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=False,
    )
    cfg.qk_norm = True
    cfg.layer_module = "Qwen2MoTDecoderLayer"   # use_moe = 'Mo' in layer_module
    cfg.freeze_und = False
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(dev).to(torch.bfloat16)


def make_inputs(dev):
    """B=1 gen geometry: query=[SOI, latents, EOI], full context-first index tensors."""
    L_q = N_LATENT + 2
    torch.manual_seed(123)
    seq = torch.randn(L_q, HIDDEN, device=dev, dtype=torch.bfloat16)
    cache = {  # replicated text context K/V per layer (requires_grad=False)
        i: (torch.randn(L_CTX, HEADS_KV, HEAD_DIM, device=dev, dtype=torch.bfloat16),
            torch.randn(L_CTX, HEADS_KV, HEAD_DIM, device=dev, dtype=torch.bfloat16))
        for i in range(N_LAYERS)
    }
    return {
        "packed_query_sequence": seq,
        "query_lens": torch.tensor([L_q], dtype=torch.int32, device=dev),
        "packed_query_position_ids": torch.arange(L_q, device=dev),
        "packed_query_indexes": torch.arange(L_CTX, L_CTX + L_q, device=dev),       # context-first
        "key_values_lens": torch.tensor([L_CTX], dtype=torch.int32, device=dev),
        "packed_key_value_indexes": torch.arange(0, L_CTX, device=dev),
        "packed_text_indexes": torch.tensor([0, L_q - 1], device=dev),              # SOI, EOI
        "packed_vae_token_indexes": torch.arange(1, L_q - 1, device=dev),           # latents
        "_cache": cache,
    }


def main():
    from unirl.models.bagel.vendor.modeling.bagel.qwen2_navit import NaiveCache

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m = build_model(dev)
    inp = make_inputs(dev)
    # Bit-identical weights+inputs across launches (seeds already make them identical
    # across ranks within a launch; the saved blob ties the sp=1 and sp=N launches).
    if os.path.exists(ST):
        blob = torch.load(ST, map_location=dev)
        m.load_state_dict(blob["sd"])
        inp = blob["inp"]
    elif rank == 0:
        torch.save({"sd": m.state_dict(), "inp": inp}, ST)
    dist.barrier()

    if world > 1:
        apply_sequence_parallelism(m, world)   # routes to the BAGEL adapter

    cache = NaiveCache(N_LAYERS)
    for i, (k, v) in inp["_cache"].items():
        cache.key_cache[i], cache.value_cache[i] = k, v
    call = {k: v for k, v in inp.items() if not k.startswith("_")}

    out = m.forward_inference(
        **call, past_key_values=cache, update_past_key_values=False,
        is_causal=False, mode="gen",
    )
    hs = out.packed_query_sequence                       # [L_q, HIDDEN] (gathered at world>1)
    loss = hs[inp["packed_vae_token_indexes"]].float().pow(2).sum()
    loss.backward()

    # full grad on a gen-expert param: replicated params -> sum slice-grads across SP
    g = m.model.layers[0].self_attn.q_proj_moe_gen.weight.grad.clone()
    if world > 1:
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=get_parallel_state().sp_group)

    if rank == 0:
        ps = get_parallel_state()
        print(f"world={world} sp_size={ps.sp_size} ulysses={ps.ulysses_enabled} "
              f"out={tuple(hs.shape)} loss={loss.item():.5f}", flush=True)
        if world == 1:
            torch.save({"hs": hs.detach().cpu(), "g": g.cpu()}, REF)
        elif os.path.exists(REF):
            ref = torch.load(REF)
            for name, a, b in (("out", ref["hs"], hs.detach().cpu()), ("grad", ref["g"], g.cpu())):
                d = (a - b).abs().max().item()
                rel = d / (a.abs().max().item() + 1e-9)
                print(f"PARITY {name} sp=1 vs sp={world}: max|Δ|={d:.3e} relerr={rel:.3e} "
                      f"{'PASS' if rel < 3e-2 else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
