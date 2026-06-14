"""flux2 diffusion Ulysses SP parity: sp=1 ref vs sp=N (model-level slice/gather).

flux2 = dual->single blocks + text-strip. Slice both streams at model input,
slice each pos_embed (cos,sin), gather image-only output at model exit. fp32->SDPA.

  torchrun --nproc_per_node=1 --master_port=29598 /tmp/sp_flux2_parity.py
  torchrun --nproc_per_node=2 --master_port=29599 /tmp/sp_flux2_parity.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism

SD = "/tmp/flux2_sd.pt"


def build(dev):
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as M
    torch.manual_seed(0)
    m = M(patch_size=1, in_channels=128, num_layers=2, num_single_layers=2,
          attention_head_dim=128, num_attention_heads=4, joint_attention_dim=128,
          guidance_embeds=False, axes_dims_rope=(32, 32, 32, 32))
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

    B, S_img, S_txt = 2, 64, 16
    torch.manual_seed(123)
    hs = torch.randn(B, S_img, 128, device=dev)
    ehs = torch.randn(B, S_txt, 128, device=dev)
    ts = torch.rand(B, device=dev)
    img_ids = torch.zeros(S_img, 4, device=dev)
    img_ids[:, 0] = torch.arange(S_img, device=dev)
    txt_ids = torch.zeros(S_txt, 4, device=dev)
    txt_ids[:, 0] = torch.arange(S_txt, device=dev)

    with torch.no_grad():
        out = m(hidden_states=hs, encoder_hidden_states=ehs, timestep=ts,
                img_ids=img_ids, txt_ids=txt_ids, return_dict=True).sample

    if rank == 0:
        print(f"world={world} sp={get_parallel_state().sp_size} ulysses={get_parallel_state().ulysses_enabled} out={tuple(out.shape)}", flush=True)
        torch.save(out.cpu(), f"/tmp/flux2_out_w{world}.pt")
        ref = "/tmp/flux2_out_w1.pt"
        if world > 1 and os.path.exists(ref):
            a, b = torch.load(ref), out.cpu()
            d = (a - b).abs().max().item()
            r = d / (a.abs().max().item() + 1e-9)
            print(f"FLUX2 PARITY sp=1 vs sp={world}: max|Δ|={d:.3e} relerr={r:.3e} {'PASS' if r < 3e-2 else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
