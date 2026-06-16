"""Is a fixed-tiling Triton matmul M-invariant? (HI3 SP alignment prototype.)

cuBLAS picks a different kernel per M, so the same per-token input gives a
different bf16 result at M=L vs M=L/sp (the SP divergence seed). A Triton matmul
with a FIXED (BM,BN,BK) tiling and a fixed K-loop reduces every row's K in the
same order regardless of M -> per-row result is M-invariant by construction.
Compare cuBLAS (M-dependent) vs Triton (should be M-invariant) on full vs sharded.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = A + om[:, None] * sam + ok[None, :] * sak
    bp = B + ok[:, None] * sbk + on[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):  # fixed K order, independent of M
        a = tl.load(ap, mask=(om[:, None] < M) & (ok[None, :] < K - k), other=0.0)
        b = tl.load(bp, mask=(ok[:, None] < K - k) & (on[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        ap += BK * sak
        bp += BK * sbk
    tl.store(C + om[:, None] * scm + on[None, :] * scn, acc.to(C.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def tmm(a, b, BM=128, BN=128, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM, BN, BK)
    return c


def run(dtype):
    TOK, H, O, SHARD = 512, 4096, 6144, 64  # full M=512, sharded into 8 x 64 (sp=8)
    torch.manual_seed(0)
    w = torch.randn(O, H, device="cuda", dtype=dtype)
    x = torch.randn(TOK, H, device="cuda", dtype=dtype)
    Bm = w.t().contiguous()  # [H, O]

    def d(full, shard):
        return (full.float() - shard.float()).abs().max().item()

    cf = F.linear(x, w)
    cs = torch.cat([F.linear(x[i:i + SHARD], w) for i in range(0, TOK, SHARD)])
    tf = tmm(x, Bm)
    ts = torch.cat([tmm(x[i:i + SHARD], Bm) for i in range(0, TOK, SHARD)])

    print(f"== dtype={dtype} (full M={TOK} vs sharded M={SHARD}) ==")
    print(f"  cuBLAS  full vs sharded : {d(cf, cs):.3e}   (the SP divergence seed)")
    print(f"  Triton  full vs sharded : {d(tf, ts):.3e}   (target: 0 -> M-invariant)")
    print(f"  Triton-full vs cuBLAS-full: {d(tf, cf):.3e}  (sanity: ~1 ULP, kernel diff)")


for dt in (torch.bfloat16, torch.float32):
    run(dt)
