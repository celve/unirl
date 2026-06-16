"""Reduce-dtype vs output-dtype: which one controls the bf16 GEMM M-dependence?

bf16 tensor-core matmul already ACCUMULATES in fp32. So the M-dependence isn't
reduction precision — it's (a) different kernels reduce K in a different ORDER
(fp32 non-associativity -> ~1e-6) and (b) the bf16 OUTPUT store rounds that to
1 ULP. Demonstrate by varying the output store dtype while M=8 vs M=64.
"""
import torch
import torch.nn.functional as F

dev = "cuda"
H, O, N = 4096, 6144, 64
torch.manual_seed(0)
w = torch.randn(O, H, device=dev, dtype=torch.bfloat16)
x = torch.randn(N, H, device=dev, dtype=torch.bfloat16)


def cmp(full, shard, tag):
    d = (full.float() - shard.float()).abs().max().item()
    r = d / (full.float().abs().max().item() + 1e-9)
    print(f"  {tag:46s}: max|Δ|={d:.3e}  rel={r:.3e}")


# 1) current: bf16 in, fp32 accumulate (default), bf16 out
f = F.linear(x, w)
s = torch.cat([F.linear(x[i : i + 8], w) for i in range(0, N, 8)])
cmp(f, s, "bf16 in, fp32 accumulate, bf16 OUT (current)")

# 2) explicitly force fp32 reduction (no split-K bf16) — still bf16 out
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
f = F.linear(x, w)
s = torch.cat([F.linear(x[i : i + 8], w) for i in range(0, N, 8)])
cmp(f, s, "bf16 in, FORCED fp32 reduce, bf16 OUT")

# 3) keep the result in fp32 (upcast inputs -> fp32 matmul -> fp32 OUT)
f = F.linear(x.float(), w.float())
s = torch.cat([F.linear(x[i : i + 8].float(), w.float()) for i in range(0, N, 8)])
cmp(f, s, "bf16 in, fp32 matmul, fp32 OUT  (keep result)")

# 4) same fp32 matmul but cast the result back to bf16 store
f = F.linear(x.float(), w.float()).bfloat16()
s = torch.cat([F.linear(x[i : i + 8].float(), w.float()).bfloat16() for i in range(0, N, 8)])
cmp(f, s, "bf16 in, fp32 matmul, bf16 OUT (re-store)")
