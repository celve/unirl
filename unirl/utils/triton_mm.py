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


def _mm_minvariant(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``a @ weight.T`` with ``a``: ``[M, K]`` contiguous, ``weight``: ``[N, K]`` (the
    row-major ``nn.Linear`` weight), via the fixed-tiling, M-invariant kernel.

    ``weight`` is read transposed *through its strides* (``B[k,n] = weight[n,k]``) — no
    transposed copy is materialized, which would otherwise allocate a full extra weight
    per call (OOM under route-all). Same values, same fixed K-reduction order, so the
    bf16 result is bit-identical to transposing first."""
    M, K = a.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 128, 128, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, weight, c, M, N, K, a.stride(0), a.stride(1),
                     weight.stride(1), weight.stride(0), c.stride(0), c.stride(1), BM, BN, BK)
    return c


class _MinvariantLinear(torch.autograd.Function):
    """Autograd-aware M-invariant linear.

    Forward goes through the fixed-tiling Triton matmul so the activation is
    M-invariant (the SP shard at M=L/sp reproduces the non-SP / generation M=L
    result bit-for-bit — what makes ``new_logp`` match ``old_logp``). Backward uses
    plain cuBLAS matmuls: the gradient only has to be a *correct* gradient of that
    forward (``grad_x = grad_out @ W``, ``grad_W = grad_out^T @ x``), it does not
    need to be M-invariant, so there is no reason to pay the Triton cost there.

    Without this, routing a grad-requiring linear through the bare kernel returns a
    tensor with no ``grad_fn`` -> "element 0 ... does not require grad" at backward.
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        shp = x.shape
        a = x.reshape(-1, shp[-1]).contiguous()
        c = _mm_minvariant(a, weight)
        # ``a`` is only needed in backward for grad_weight. For a frozen weight (the
        # LoRA-frozen base — most linears) it is never needed, so don't pin it: this
        # matches cuBLAS F.linear, which saves no input for a frozen linear. Saving it
        # unconditionally pinned every base activation -> OOM under route-all.
        ctx.save_for_backward(a if weight.requires_grad else None, weight)
        ctx.shp = shp
        out = c.reshape(*shp[:-1], weight.shape[0])
        return out + bias if bias is not None else out

    @staticmethod
    def backward(ctx, grad_out):
        a, weight = ctx.saved_tensors          # a: [M, K], weight: [N, K]
        n = weight.shape[0]
        go = grad_out.reshape(-1, n)           # [M, N]
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = (go @ weight).reshape(ctx.shp)        # [M,N] @ [N,K] -> [M,K]
        if ctx.needs_input_grad[1]:
            grad_w = go.t() @ a                            # [N,M] @ [M,K] -> [N,K]
        if ctx.needs_input_grad[2]:
            grad_b = go.sum(0)
        return grad_x, grad_w, grad_b


def minvariant_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` (``x @ weight.T + bias``) via the fixed-tiling, M-invariant kernel.

    ``x``: ``[..., in]``, ``weight``: ``[out, in]``. Differentiable (see
    ``_MinvariantLinear``). Falls back nowhere — call only for the shapes/dtypes you
    want made M-invariant (the caller gates on M/dtype).
    """
    return _MinvariantLinear.apply(x, weight, bias)


__all__ = ["minvariant_linear"]
