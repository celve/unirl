"""Wan diffusion Ulysses SP parity: sp=1 ref vs sp=N (dispatch-patch + cross-attn).

Wan = image self-attention (all-to-all) + text cross-attention (full text K/V,
no all-to-all via the dispatch guard). 5D latent input, Wan 4D RoPE. fp32->SDPA.

  torchrun --nproc_per_node=1 --master_port=29594 /tmp/sp_wan_parity.py
  torchrun --nproc_per_node=2 --master_port=29595 /tmp/sp_wan_parity.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism

SD = "/tmp/wan_sd.pt"


def build(dev):
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as M
    torch.manual_seed(0)
    m = M(
        patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=128,
        in_channels=16, out_channels=16, text_dim=128, freq_dim=256,
        num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
    )
    return m.to(dev).to(torch.float32)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m = build(dev)
    if os.path.exists(SD):
        m.load_state_dict(torch.load(SD, map_location=dev))
    elif rank == 0:
        torch.save(m.state_dict(), SD)
    dist.barrier()
    apply_sequence_parallelism(m, world)  # no-op at world=1
    m.eval()

    B = 2  # latent (B,C,T,H,W); S_img = T*(H/2)*(W/2) = 4*8*8 = 256 (div by sp)
    torch.manual_seed(123)
    hs = torch.randn(B, 16, 4, 16, 16, device=dev)
    ehs = torch.randn(B, 16, 128, device=dev)   # (B, S_txt, text_dim) — full text (cross-attn)
    ts = torch.rand(B, device=dev) * 1000

    with torch.no_grad():
        out = m(hidden_states=hs, encoder_hidden_states=ehs, timestep=ts, return_dict=True).sample

    if rank == 0:
        print(f"world={world} sp={get_parallel_state().sp_size} ulysses={get_parallel_state().ulysses_enabled} out={tuple(out.shape)}", flush=True)
        torch.save(out.cpu(), f"/tmp/wan_out_w{world}.pt")
        ref = "/tmp/wan_out_w1.pt"
        if world > 1 and os.path.exists(ref):
            a, b = torch.load(ref), out.cpu()
            d = (a - b).abs().max().item()
            r = d / (a.abs().max().item() + 1e-9)
            print(f"WAN PARITY sp=1 vs sp={world}: max|Δ|={d:.3e} relerr={r:.3e} {'PASS' if r < 3e-2 else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
