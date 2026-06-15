"""Diffusion SP-vs-FSDP peak-memory bench: SD3.5-medium transformer, FSDP2 over the
folded mesh, sp=1 (FSDP baseline) vs sp=2 (FSDP + Ulysses). SDPA attention (no
flash-kernel-hub dependency). Measures torch.cuda.max_memory_allocated for fwd+bwd
at increasing image resolution (= joint-attention sequence length).

  SP=1 HLAT=128 torchrun --nproc_per_node=2 --master_port=296XX sd3_mem_fsdp.py
  SP=2 HLAT=128 torchrun --nproc_per_node=2 --master_port=296YY sd3_mem_fsdp.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism
from torch.distributed.fsdp import fully_shard

SP = int(os.environ.get("SP", "1"))
HLAT = int(os.environ.get("HLAT", "128"))   # latent H=W; image tokens = (HLAT/2)^2
LAYERS = int(os.environ.get("LAYERS", "24"))
S_TXT = int(os.environ.get("S_TXT", "256"))  # encoder (text) tokens; even -> sp-divisible


def build(dev):
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as M
    torch.manual_seed(0)
    m = M(
        sample_size=HLAT, patch_size=2, in_channels=16, out_channels=16, num_layers=LAYERS,
        attention_head_dim=64, num_attention_heads=24, joint_attention_dim=4096,
        caption_projection_dim=1536, pooled_projection_dim=2048, pos_embed_max_size=192,
    )
    return m.to(dev).to(torch.bfloat16)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=world // SP, ulysses_size=SP, dp_mode="fsdp2", device_type="cuda")
    ps = get_parallel_state()

    m = build(dev)
    mesh = ps.fsdp_mesh
    for blk in m.transformer_blocks:
        fully_shard(blk, mesh=mesh)
    if SP > 1:
        apply_sequence_parallelism(m, SP)
    m.train()

    B = 1
    H = HLAT
    img_tokens = (H // 2) ** 2
    hs = torch.randn(B, 16, H, H, device=dev, dtype=torch.bfloat16)
    ehs = torch.randn(B, S_TXT, 4096, device=dev, dtype=torch.bfloat16)
    pooled = torch.randn(B, 2048, device=dev, dtype=torch.bfloat16)
    ts = (torch.rand(B, device=dev) * 1000).to(torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ok, peak = True, 0.0
    try:
        out = m(hidden_states=hs, encoder_hidden_states=ehs, pooled_projections=pooled,
                timestep=ts, return_dict=True).sample
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        ok = False
    if rank == 0:
        seq = img_tokens + S_TXT
        print(f"SD3MEM sp={SP} world={world} latent={H} seq={seq} per_rank_seq~{seq // SP} "
              f"peak={'OOM' if not ok else f'{peak:.2f}GB'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
