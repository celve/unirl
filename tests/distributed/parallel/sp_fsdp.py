"""R1 confirmation with REAL FSDP2: fully_shard over the folded dp_shard_sp mesh.

Verifies that FSDP2's reduce-scatter (averaging over the folded mesh, which
includes the ulysses dim) yields the correct full-sequence gradient with NO
manual sp_size compensation. Run: torchrun --nproc_per_node=2 /tmp/sp_fsdp.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.distributed.sequence_parallel import slice_input_tensor, gather_outputs
from veomni.distributed.sequence_parallel.ulysses import gather_seq_scatter_heads, gather_heads_scatter_seq
from torch.distributed.fsdp import fully_shard

B, L, H, Dh = 2, 16, 4, 16
D = H * Dh


def make_model(dev):
    torch.manual_seed(1234)
    return nn.ModuleDict(dict(
        q=nn.Linear(D, D, bias=False), k=nn.Linear(D, D, bias=False),
        v=nn.Linear(D, D, bias=False), o=nn.Linear(D, D, bias=False),
    )).to(dev).to(torch.float32)


def attn_full(m, x):
    Bn, Ln, _ = x.shape
    q = m['q'](x).view(Bn, Ln, H, Dh).transpose(1, 2)
    k = m['k'](x).view(Bn, Ln, H, Dh).transpose(1, 2)
    v = m['v'](x).view(Bn, Ln, H, Dh).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, Ln, D)
    return m['o'](out)


def attn_sp(m, x_local, spg):
    Bn, Ll, _ = x_local.shape
    q = m['q'](x_local).view(Bn, Ll, H, Dh)
    k = m['k'](x_local).view(Bn, Ll, H, Dh)
    v = m['v'](x_local).view(Bn, Ll, H, Dh)
    q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=spg)
    k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, group=spg)
    v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, group=spg)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    out = gather_heads_scatter_seq(out, head_dim=2, seq_dim=1, group=spg)
    return m['o'](out.reshape(Bn, Ll, D))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")
    ps = get_parallel_state()
    spg = ps.sp_group
    fsdp_mesh = ps.fsdp_mesh
    if rank == 0:
        print(f"sp_size={ps.sp_size}  fsdp_mesh={fsdp_mesh}", flush=True)

    torch.manual_seed(42)
    x_full = torch.randn(B, L, D, device=dev, dtype=torch.float32)

    # ground-truth reference: unsharded, full sequence, local (no comm)
    m_ref = make_model(dev)
    attn_full(m_ref, x_full).float().pow(2).mean().backward()
    g_ref = {n: p.grad.detach().clone() for n, p in m_ref.named_parameters()}

    # real FSDP2 over the folded mesh + Ulysses SP
    m_sp = make_model(dev)
    for sub in m_sp.values():
        fully_shard(sub, mesh=fsdp_mesh)
    x_local = slice_input_tensor(x_full, dim=1, group=spg)
    out_full = gather_outputs(attn_sp(m_sp, x_local, spg), gather_dim=1, group=spg)
    out_full.float().pow(2).mean().backward()

    if rank == 0:
        print("[grad parity: FSDP2 full_tensor(grad) vs unsharded ref]", flush=True)
    for n, p in m_sp.named_parameters():
        g = p.grad
        g = g.full_tensor() if hasattr(g, "full_tensor") else g
        if rank == 0:
            den = g_ref[n].abs().max().item() + 1e-12
            relerr = (g - g_ref[n]).abs().max().item() / den
            print(f"  {n:>14}: relerr={relerr:.3e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
