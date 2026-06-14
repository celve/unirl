"""R1 de-risk: VeOmni Ulysses slice/gather forward+grad parity vs single-proc reference.

Run: torchrun --nproc_per_node=2 /tmp/sp_parity.py
Isolates the sp_size gradient-compensation question (no FSDP; params replicated,
cross-SP grad combined manually with SUM and MEAN to mimic the two FSDP regimes).
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Use unirl's shim so veomni/__init__ _apply_patches does NOT run.
from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.distributed.sequence_parallel import slice_input_tensor, gather_outputs
from veomni.distributed.sequence_parallel.ulysses import (
    gather_seq_scatter_heads,
    gather_heads_scatter_seq,
)

B, L, H, Dh = 2, 16, 4, 16
D = H * Dh


def make_model(dev):
    torch.manual_seed(1234)  # identical weights on every rank
    return nn.ModuleDict(dict(
        q=nn.Linear(D, D, bias=False),
        k=nn.Linear(D, D, bias=False),
        v=nn.Linear(D, D, bias=False),
        o=nn.Linear(D, D, bias=False),
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
    # Ulysses: [B, L/sp, H, Dh] -> [B, L, H/sp, Dh]
    q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=spg)
    k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, group=spg)
    v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, group=spg)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    out = out.transpose(1, 2)  # [B, L, H/sp, Dh]
    out = gather_heads_scatter_seq(out, head_dim=2, seq_dim=1, group=spg)  # [B, L/sp, H, Dh]
    return m['o'](out.reshape(Bn, Ll, D))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    SP = world
    init_parallel_state(dp_size=1, ulysses_size=SP, dp_mode="fsdp2", device_type="cuda")
    ps = get_parallel_state()
    spg = ps.sp_group
    if rank == 0:
        print(f"world={world} sp_size={ps.sp_size} sp_enabled={ps.sp_enabled} ulysses_enabled={ps.ulysses_enabled}", flush=True)
    assert ps.sp_size == SP

    torch.manual_seed(42)
    x_full = torch.randn(B, L, D, device=dev, dtype=torch.float32)  # identical on all ranks

    # reference: full sequence, no SP, computed locally (no comm) -> ground truth
    m_ref = make_model(dev)
    loss_ref = attn_full(m_ref, x_full).float().pow(2).mean()
    loss_ref.backward()
    g_ref = {n: p.grad.detach().clone() for n, p in m_ref.named_parameters()}

    # SP path
    m_sp = make_model(dev)
    x_local = slice_input_tensor(x_full, dim=1, group=spg)
    out_local = attn_sp(m_sp, x_local, spg)
    out_full = gather_outputs(out_local, gather_dim=1, group=spg)
    loss_sp = out_full.float().pow(2).mean()
    loss_sp.backward()

    g_sum, g_mean = {}, {}
    for n, p in m_sp.named_parameters():
        gs = p.grad.detach().clone()
        dist.all_reduce(gs, op=dist.ReduceOp.SUM, group=spg)
        g_sum[n] = gs
        g_mean[n] = gs / SP

    if rank == 0:
        fwd = (out_full - attn_full(m_ref, x_full)).abs().max().item()
        print(f"[fwd] max|out_sp-out_ref|={fwd:.3e}  loss_ref={loss_ref.item():.6f} loss_sp={loss_sp.item():.6f}", flush=True)
        for n in g_ref:
            den = g_ref[n].abs().max().item() + 1e-12
            e_sum = (g_sum[n] - g_ref[n]).abs().max().item() / den
            e_mean = (g_mean[n] - g_ref[n]).abs().max().item() / den
            print(f"[grad {n:>14}] relerr(SUM)={e_sum:.3e}  relerr(MEAN)={e_mean:.3e}", flush=True)
        print("VERDICT: SUM-combine correct => FSDP needs *sp_size comp; "
              "MEAN-combine correct => folded FSDP avg is already right.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
