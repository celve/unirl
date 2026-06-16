"""Is a bf16 Linear (cuBLAS GEMM) M-dimension-dependent? (HI3 SP root-cause.)

Under Ulysses SP each rank runs the per-token projections on L/sp tokens, while
the non-SP reference runs them on L tokens. If cuBLAS picks a different GEMM
kernel / K-reduction order for a smaller M, the SAME per-token input yields a
different bf16 output -> the seed of the SP-vs-nonSP forward divergence. Compare
lin(x) computed on the full batch vs the concatenation of per-shard sub-batches.
"""
import torch

torch.manual_seed(0)
dev = "cuda"
H, O, N = 4096, 6144, 64  # qkv_proj-like dims; N tokens (e.g. seq len), sharded by 8


def probe(dtype):
    lin = torch.nn.Linear(H, O, bias=False).to(dev).to(dtype)
    x = torch.randn(N, H, device=dev, dtype=dtype)
    with torch.no_grad():
        y_full = lin(x)  # M = N
        print(f"== dtype={dtype} (full M={N}) ==")
        for sp in (2, 4, 8):
            m = N // sp
            y_shard = torch.cat([lin(x[i : i + m]) for i in range(0, N, m)], dim=0)  # M = N/sp per call
            diff = (y_full.float() - y_shard.float()).abs().max().item()
            rel = diff / (y_full.float().abs().max().item() + 1e-9)
            print(f"   sp={sp} (M={m}): full vs sharded  max|Δ|={diff:.3e}  rel={rel:.3e}")


for dt in (torch.bfloat16, torch.float32):
    probe(dt)
