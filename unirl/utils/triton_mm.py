"""M-invariant matmul (fixed-tiling Triton) for HI3 sequence-parallel training.

bf16 cuBLAS picks a different kernel per M (number of rows), so the same per-token
input yields a different bf16 result at M=L (non-SP / generation) vs M=L/sp (SP) —
the source of the SP GRPO-ratio divergence (see docs / sp_hi3_bisect.py). A matmul
with a FIXED (BM,BN,BK) tiling reduces every row's K in the same order regardless
of M, so its per-row result is M-invariant. Empirically, in bf16 it reproduces the
cuBLAS large-M result exactly (the Triton-vs-cuBLAS fp32 gap is below the bf16 ULP),
so routing the small-M GEMMs (MoE experts + small projections) through it makes the
SP train forward bit-match the non-SP/generation forward, fixing the ratio while
keeping Ulysses + bf16 and leaving old_logp untouched.
"""
from __future__ import annotations

import torch
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
    for k in range(0, K, BK):  # fixed K order, independent of M -> M-invariant
        a = tl.load(ap, mask=(om[:, None] < M) & (ok[None, :] < K - k), other=0.0)
        b = tl.load(bp, mask=(ok[:, None] < K - k) & (on[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        ap += BK * sak
        bp += BK * sbk
    tl.store(C + om[:, None] * scm + on[None, :] * scn, acc.to(C.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def minvariant_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` (``x @ weight.T + bias``) via the fixed-tiling, M-invariant kernel.

    ``x``: ``[..., in]``, ``weight``: ``[out, in]``. Falls back nowhere — call only
    for the shapes/dtypes you want made M-invariant (the caller gates on M/dtype).
    """
    shp = x.shape
    a = x.reshape(-1, shp[-1]).contiguous()
    b = weight.t().contiguous()
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 128, 128, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM, BN, BK)
    c = c.reshape(*shp[:-1], N)
    return c + bias if bias is not None else c


__all__ = ["minvariant_linear"]
