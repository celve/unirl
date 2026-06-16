"""Can a bf16 Linear be made M-invariant? (HI3 SP alignment-without-replay test.)

The SP-vs-nonSP forward divergence is bf16 cuBLAS picking a different kernel /
K-reduction order for M=L/sp vs M=L. Test whether forcing an M-invariant
reduction (no split-K / fp32 reduction / deterministic algos) collapses it.
Run with CUBLAS_WORKSPACE_CONFIG=:4096:8 so deterministic mode is allowed.
"""
import torch

dev = "cuda"
H, O, N = 4096, 6144, 64  # qkv_proj-like; N tokens sharded by 8 -> M=8 vs M=64


def measure(tag, dtype=torch.bfloat16):
    torch.manual_seed(0)
    lin = torch.nn.Linear(H, O, bias=False).to(dev).to(dtype)
    x = torch.randn(N, H, device=dev, dtype=dtype)
    with torch.no_grad():
        y_full = lin(x)  # M=64
        y_shard = torch.cat([lin(x[i : i + 8]) for i in range(0, N, 8)], dim=0)  # M=8
    diff = (y_full.float() - y_shard.float()).abs().max().item()
    rel = diff / (y_full.float().abs().max().item() + 1e-9)
    print(f"  {tag:48s}: max|Δ|={diff:.3e}  rel={rel:.3e}")


print("== baseline ==")
measure("bf16 default")
measure("fp32 default", torch.float32)

print("== knob: bf16 reduced-precision reduction OFF ==")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
measure("bf16 allow_reduced_precision_reduction=False")

print("== knob: deterministic algorithms ==")
try:
    torch.use_deterministic_algorithms(True)
    measure("bf16 deterministic + reduced_reduction=False")
except Exception as e:
    print(f"  deterministic mode ERR: {str(e)[:90]}")
