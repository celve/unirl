"""AR (Qwen3) Ulysses SP logprob parity: sp=1 reference vs sp=N.

Validates the real integration — VeOmni registered SP attention + the decoder
slice/gather wrapper (unirl.train.backend.veomni.sp) — on a small real Qwen3.

Both runs use the SAME attention kernel (veomni_flash_attention_2_with_sp; at
sp=1 it is plain flash with the all-to-all disabled), so the only difference is
the Ulysses decomposition. Run sp=1 first (saves weights + ref), then sp=N:

  torchrun --nproc_per_node=1 --master_port=29570 /tmp/sp_ar_parity.py   # ref
  torchrun --nproc_per_node=2 --master_port=29571 /tmp/sp_ar_parity.py   # compare

Residual diff is bf16 attention reduction-order (full heads vs sharded heads),
expected ~1e-2; a real SP bug shows up as a much larger, systematic diff.
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
_compat.ensure_attention_patch_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism
from unirl.train.backend.veomni.sp.ar import SP_ATTN_IMPL

SD_PATH = "/tmp/ar_sd.pt"


def build_model(dev):
    from transformers import AutoModelForCausalLM, Qwen3Config
    cfg = Qwen3Config(
        vocab_size=1000, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=512, head_dim=32,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(dev).to(torch.bfloat16)


def set_attn(m):
    m.config._attn_implementation = SP_ATTN_IMPL
    for mod in m.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = SP_ATTN_IMPL


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m = build_model(dev)
    # share identical weights across launches via the saved state dict
    if os.path.exists(SD_PATH):
        m.load_state_dict(torch.load(SD_PATH, map_location=dev))
    elif rank == 0:
        torch.save(m.state_dict(), SD_PATH)
    dist.barrier()

    set_attn(m)                       # same kernel for both worlds
    if world > 1:
        apply_sequence_parallelism(m, world)   # installs decoder slice/gather wrapper
    m.eval()

    B, L = 2, 64
    torch.manual_seed(123)
    ids = torch.randint(0, 1000, (B, L), device=dev)
    pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, L).contiguous()

    with torch.no_grad():
        hidden = m.model(input_ids=ids, attention_mask=None, position_ids=pos,
                         use_cache=False, return_dict=True).last_hidden_state
        logits = m.lm_head(hidden).float()
        logp = logits.log_softmax(-1)
        chosen = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    if rank == 0:
        print(f"world={world} sp_size={get_parallel_state().sp_size} "
              f"ulysses={get_parallel_state().ulysses_enabled} chosen={tuple(chosen.shape)}", flush=True)
        torch.save(chosen.cpu(), f"/tmp/ar_logp_w{world}.pt")
        ref_path = "/tmp/ar_logp_w1.pt"
        if world > 1 and os.path.exists(ref_path):
            a, b = torch.load(ref_path), chosen.cpu()
            diff = (a - b).abs().max().item()
            relerr = diff / (a.abs().max().item() + 1e-9)
            print(f"PARITY sp=1 vs sp={world}: max|Δlogp|={diff:.3e} relerr={relerr:.3e} "
                  f"{'PASS' if relerr < 3e-2 else 'CHECK'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
