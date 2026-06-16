"""Grouped-GEMM MoE for HI3 sequence-parallel training — M-invariant AND bit-exact.

The SP GRPO-ratio bug is the bf16 cuBLAS M-dependence in the MoE expert GEMMs: at
M=L/sp (SP shard) they diverge from M=L (non-SP / generation), the 1-ULP seed
compounds over layers, flips top-k routing, and the ratio breaks.

VeOmni's grouped-GEMM kernel (``group_gemm_same_nk``, fixed ``BLOCK_K``) is BOTH
M-invariant and **bit-exact to cuBLAS-at-M=full** on these shapes/data (verified 0%
element divergence on real activations) — at ~cuBLAS speed (one fused launch for all
experts). VeOmni's *fused-MoE wrapper* is NOT bit-exact (it applies the router weight
before fc2 and uses its own scatter/gather, which bf16-round differently than HI3), so
we drive the kernel directly with HI3's EXACT op order:

    repeat_interleave(topk) -> argsort by expert -> group_gemm(gate_up) ->
    x1*act(x2) -> group_gemm(down) -> unsort -> (*topk_weight).sum(topk)   # weight AFTER down

Experts are restructured to stacked frozen params (``moe_gate_up`` [E,2I,H] in HI3's
native ``[up; gate]`` order, ``moe_down`` [E,H,I]) so there is no per-forward stacking
copy; the 80B checkpoint per-expert keys are remapped at load (bundle.materialize).
The router gate (fp32) + shared MLP stay on F.linear (gate -> pad-to-M=L by the SP
adapter); qkv/o_proj stay on cuBLAS.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

_HI3_MOE_CLASS = "HunyuanMoE"


def pin_group_gemm_block_k() -> None:
    """Force VeOmni's grouped-GEMM to one fixed config (CATCH_ALL fallback, BLOCK_K=32)
    for all ``total_M`` so the K-reduction order is M-invariant (its pretuned algo-key
    buckets on ``total_M//5000``; under SP the shard would land in a different bucket and
    could pick a different ``BLOCK_K``). Idempotent."""
    try:
        from veomni.ops.kernels.moe._kernels.kernel import group_gemm as _gg
        from veomni.ops.kernels.moe._kernels.utils.pretuned import CATCH_ALL_ALGO_KEY
    except Exception:
        return

    def _const(**_kw):
        return CATCH_ALL_ALGO_KEY

    for kern in (_gg.group_gemm_same_nk_kernel, _gg.group_gemm_same_mn_kernel):
        if hasattr(kern, "algo_key_maker") and not getattr(kern, "_unirl_pinned", False):
            kern.algo_key_maker = _const
            kern._unirl_pinned = True  # type: ignore[attr-defined]


class _GroupedLinear(torch.autograd.Function):
    """Autograd-aware grouped linear ``out[t] = a[t] @ w[e(t)].T`` over expert groups
    (``cumsum`` = inclusive token-count prefix sum). Forward + dgrad via the bit-exact
    ``group_gemm_same_nk``; wgrad via ``group_gemm_same_mn`` only if ``w`` needs grad
    (HI3 experts are LoRA-frozen, so wgrad is skipped and ``a`` is not pinned)."""

    @staticmethod
    def forward(ctx, a, w, cumsum):
        from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_nk

        out = group_gemm_same_nk(a, w, cumsum, a.shape[0], transpose_b=True)
        ctx.save_for_backward(a if w.requires_grad else None, w, cumsum)
        return out

    @staticmethod
    def backward(ctx, go):
        from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk

        a, w, cumsum = ctx.saved_tensors
        go = go.contiguous()
        grad_a = grad_w = None
        if ctx.needs_input_grad[0]:
            # dgrad: go @ w[e]  (w [E,N,K] read as [E, K_b=N, N_b=K] with transpose_b=False)
            grad_a = group_gemm_same_nk(go, w, cumsum, go.shape[0], transpose_b=False)
        if ctx.needs_input_grad[1]:
            grad_w = torch.empty_like(w)
            group_gemm_same_mn(go, a, grad_w, cumsum, go.shape[0], transpose_a=True, transpose_b=False)
        return grad_a, grad_w, None


def _grouped_linear(a, w, cumsum):
    return _GroupedLinear.apply(a.contiguous(), w, cumsum)


def restructure_hi3_experts(model: nn.Module) -> int:
    """Replace every ``HunyuanMoE.experts`` ModuleList with stacked frozen params
    ``moe_gate_up`` [E,2I,H] (HI3 native ``[up; gate]`` order) + ``moe_down`` [E,H,I].
    Works on meta (shapes only; data filled by the checkpoint remap) or materialized
    (copies data). Frees the per-expert modules. Idempotent. Returns #blocks changed."""
    n = 0
    for moe in model.modules():
        if type(moe).__name__ != _HI3_MOE_CLASS or getattr(moe, "_grouped", False):
            continue
        experts = moe.experts
        E = len(experts)
        w_gu = experts[0].gate_and_up_proj.weight   # [2I, H]
        w_dn = experts[0].down_proj.weight           # [H, I]
        assert experts[0].gate_and_up_proj.bias is None and experts[0].down_proj.bias is None, (
            "HI3 grouped MoE assumes mlp_bias=False"
        )
        rg = w_gu.requires_grad
        if w_gu.is_meta:
            gu = torch.empty(E, *w_gu.shape, device="meta", dtype=w_gu.dtype)
            dn = torch.empty(E, *w_dn.shape, device="meta", dtype=w_dn.dtype)
        else:
            with torch.no_grad():
                gu = torch.stack([e.gate_and_up_proj.weight for e in experts]).contiguous()
                dn = torch.stack([e.down_proj.weight for e in experts]).contiguous()
        moe.register_parameter("moe_gate_up", nn.Parameter(gu, requires_grad=rg))
        moe.register_parameter("moe_down", nn.Parameter(dn, requires_grad=rg))
        moe._act_fn = experts[0].act_fn            # type: ignore[attr-defined]
        del moe.experts
        moe._grouped = True                         # type: ignore[attr-defined]
        moe._n_experts = E                          # type: ignore[attr-defined]
        n += 1
    return n


def _grouped_moe_forward(self, hidden_states):  # noqa: ANN001
    """HI3 ``HunyuanMoE.forward`` driven by the bit-exact grouped kernel, op-order
    identical to HI3's per-expert path (so it bit-matches cuBLAS / the rollout)."""
    bsz, seq, H = hidden_states.shape
    shared = self.shared_mlp(hidden_states) if self.config.use_mixed_mlp_moe else None
    with torch.autocast("cuda", enabled=False):
        topk_w, topk_i = self.gate(hidden_states, topk_impl="easy")
    topk_w = topk_w.to(hidden_states.dtype)
    topk = self.moe_topk if isinstance(self.moe_topk, int) else self.moe_topk
    E = self._n_experts

    flat = hidden_states.reshape(-1, H)
    repeated = flat.repeat_interleave(topk, dim=0)               # [T0*topk, H]
    fi = topk_i.reshape(-1)
    order = fi.argsort(stable=True)
    inv_order = order.argsort()
    scattered = repeated[order]                                  # grouped by expert
    cumsum = torch.bincount(fi, minlength=E).cumsum(0).to(torch.int32)

    gu = _grouped_linear(scattered, self.moe_gate_up, cumsum)    # [T, 2I]
    x1, x2 = gu.chunk(2, dim=-1)                                 # HI3: x1=up, x2=gate
    act = x1 * self._act_fn(x2)
    dn = _grouped_linear(act, self.moe_down, cumsum)             # [T, H]

    eo = dn[inv_order]                                           # unsort -> token order
    combined = (eo.view(-1, topk, H) * topk_w.reshape(-1, topk, 1)).sum(dim=1)
    out = combined.view(bsz, seq, H)
    return shared + out if shared is not None else out


def patch_grouped_moe_forward(model: nn.Module) -> None:
    """Point ``HunyuanMoE.forward`` at the grouped path + pin ``BLOCK_K``. Run AFTER
    ``fully_shard`` (restructure must already have run). Idempotent."""
    pin_group_gemm_block_k()
    for m in model.modules():
        cls = type(m)
        if cls.__name__ == _HI3_MOE_CLASS and not getattr(cls, "_unirl_grouped", False):
            cls.forward = _grouped_moe_forward
            cls._unirl_grouped = True  # type: ignore[attr-defined]
            break


def patch_gathered_moe_forward(model: nn.Module) -> None:
    """SP isolation / reference fix: run each ``HunyuanMoE`` on the FULL (gathered)
    sequence so every expert sees its true ``M_e^full`` -> per-expert ``cuBLAS@M_e^full``
    == the no-SP baseline, bit-exact. This eliminates the only residual M-dependence (the
    grouped/large-M kernel disagrees with the per-M cuBLAS baseline on low-traffic
    experts, which compounds over layers and breaks the GRPO ratio).

    Gathers the MoE input across the sp group, runs the ORIGINAL per-expert forward at
    full L, slices the output back to the local shard. The MoE is frozen (LoRA targets
    only qkv/o_proj) and token-independent, so the redundant full-L compute is
    gradient-correct (no cross-rank / no wgrad double-count). Costs MoE compute/memory
    sharding under SP — EP (route tokens to expert owners) is the perf-equivalent
    follow-up that keeps this exact numerics. Idempotent; a runtime no-op at sp<=1.

    Requires the experts to be the ORIGINAL per-expert ModuleList (do NOT restructure to
    the grouped/large-M path, which would defeat the point).
    """
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

    for m in model.modules():
        cls = type(m)
        if cls.__name__ != _HI3_MOE_CLASS or getattr(cls, "_unirl_gathered", False):
            continue
        orig = cls.forward

        @functools.wraps(orig)
        def _gathered(self, hidden_states, _orig=orig):  # noqa: ANN001
            ps = get_parallel_state()
            if not ps.ulysses_enabled or int(getattr(ps, "ulysses_size", 1)) <= 1:
                return _orig(self, hidden_states)
            spg = ps.sp_group
            full = gather_outputs(hidden_states, gather_dim=1, group=spg)  # [B, L, H]
            # The full-L MoE is sp x the sharded activation; recompute it in backward
            # (only the small [B,L,H] input is kept) so the redundant full-seq compute
            # does not blow up the forward peak. Frozen experts -> dgrad-only recompute.
            if torch.is_grad_enabled() and hidden_states.requires_grad:
                out = torch.utils.checkpoint.checkpoint(_orig, self, full, use_reentrant=False)
            else:
                out = _orig(self, full)                                    # cuBLAS @ M_e^full
            return slice_input_tensor(out, dim=1, group=spg)               # back to local shard

        cls.forward = _gathered
        cls._unirl_gathered = True  # type: ignore[attr-defined]


__all__ = [
    "restructure_hi3_experts",
    "patch_grouped_moe_forward",
    "patch_gathered_moe_forward",
    "pin_group_gemm_block_k",
]
