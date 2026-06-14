"""Diffusion SP BACKWARD parity (qwen-image): sp=1 grads vs sp=N MEAN-combined.

Confirms the diffusion SP autograd path: block0 slices a NON-leaf (post-embed)
tensor, the dispatch all-to-all (SeqAllToAll) + gather_outputs are autograd-aware,
and FSDP's mean-over-folded-mesh is the right combiner (per docs/usp-derisk/sp_fsdp.py).
No FSDP here; grads MEAN-combined across the SP group to emulate it.

  torchrun --nproc_per_node=1 --master_port=29596 /tmp/sp_diffusion_backward.py
  torchrun --nproc_per_node=2 --master_port=29597 /tmp/sp_diffusion_backward.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism

SD = "/tmp/qib_sd.pt"


def build(dev):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as M
    torch.manual_seed(0)
    m = M(patch_size=2, in_channels=64, out_channels=16, num_layers=2,
          attention_head_dim=32, num_attention_heads=4, joint_attention_dim=128,
          guidance_embeds=False, axes_dims_rope=(8, 12, 12))
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
    m.train()

    B, S_img, S_txt = 2, 64, 16
    torch.manual_seed(123)
    hs = torch.randn(B, S_img, 64, device=dev)
    ehs = torch.randn(B, S_txt, 128, device=dev)
    ts = torch.rand(B, device=dev)
    out = m(hidden_states=hs, encoder_hidden_states=ehs, encoder_hidden_states_mask=None,
            timestep=ts, img_shapes=[[(1, 8, 8)]] * B, txt_seq_lens=[S_txt] * B, return_dict=True).sample
    out.float().pow(2).mean().backward()

    # collect grads; MEAN-combine across SP (emulates FSDP folded-mesh reduce)
    grads = {}
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().clone()
        if world > 1:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=get_parallel_state().sp_group)
            g /= world
        grads[n] = g.cpu()

    if rank == 0:
        torch.save(grads, f"/tmp/qib_grad_w{world}.pt")
        print(f"world={world} sp={get_parallel_state().sp_size} nparams_with_grad={len(grads)}", flush=True)
        ref = "/tmp/qib_grad_w1.pt"
        if world > 1 and os.path.exists(ref):
            r = torch.load(ref)
            worst = 0.0
            for n in r:
                den = r[n].abs().max().item() + 1e-9
                e = (grads[n] - r[n]).abs().max().item() / den
                worst = max(worst, e)
            print(f"DIFFUSION BACKWARD PARITY sp=1 vs sp={world}: worst relerr={worst:.3e} "
                  f"{'PASS' if worst < 3e-2 else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
