"""SD3 diffusion Ulysses SP parity: sp=1 ref vs sp=N (processor-injection path).

Validates the SPAttentionProcessor injection + sd3 boundary hooks on a tiny real
SD3Transformer2DModel (no RoPE; JointAttnProcessor2_0). fp32 -> SDPA both runs.

  torchrun --nproc_per_node=1 --master_port=29590 /tmp/sp_sd3_parity.py
  torchrun --nproc_per_node=2 --master_port=29591 /tmp/sp_sd3_parity.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism

SD = "/tmp/sd3_sd.pt"


def build(dev):
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as M
    torch.manual_seed(0)
    m = M(
        sample_size=16, patch_size=2, in_channels=16, out_channels=16, num_layers=2,
        attention_head_dim=32, num_attention_heads=4, joint_attention_dim=128,
        caption_projection_dim=128, pooled_projection_dim=64, pos_embed_max_size=16,
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

    B, H, W, S_txt = 2, 16, 16, 16  # patchified img = (16/2)^2 = 64 tokens (div by sp)
    torch.manual_seed(123)
    hs = torch.randn(B, 16, H, W, device=dev)
    ehs = torch.randn(B, S_txt, 128, device=dev)
    pooled = torch.randn(B, 64, device=dev)
    ts = torch.rand(B, device=dev)

    with torch.no_grad():
        out = m(hidden_states=hs, encoder_hidden_states=ehs, pooled_projections=pooled,
                timestep=ts, return_dict=True).sample

    if rank == 0:
        print(f"world={world} sp={get_parallel_state().sp_size} ulysses={get_parallel_state().ulysses_enabled} out={tuple(out.shape)}", flush=True)
        torch.save(out.cpu(), f"/tmp/sd3_out_w{world}.pt")
        ref = "/tmp/sd3_out_w1.pt"
        if world > 1 and os.path.exists(ref):
            a, b = torch.load(ref), out.cpu()
            d = (a - b).abs().max().item()
            r = d / (a.abs().max().item() + 1e-9)
            print(f"SD3 PARITY sp=1 vs sp={world}: max|Δ|={d:.3e} relerr={r:.3e} {'PASS' if r < 3e-2 else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
