"""Grouped-GEMM MoE for HI3 sequence-parallel training (M-invariant experts).

The SP GRPO-ratio bug is the bf16 cuBLAS M-dependence in the MoE expert GEMMs: at
M=L/sp (SP shard) the experts diverge from M=L (non-SP / generation), the 1-ULP
seed compounds over layers and flips top-k routing, and the ratio breaks. VeOmni's
grouped-GEMM (``group_gemm_same_nk`` with a fixed ``BLOCK_K``) is M-invariant *and*
reproduces cuBLAS-at-M=full, at ~cuBLAS speed (one fused launch for all experts vs
64 per-expert matmuls).

This module:
  * ``restructure_hi3_experts`` — replaces each ``HunyuanMoE.experts`` (ModuleList of
    per-expert ``HunyuanMLP``) with stacked frozen params ``moe_gate_up`` [E, 2I, H]
    (in ``[gate; up]`` order) and ``moe_down`` [E, H, I]. Memory-neutral (frees the
    per-expert modules); run on meta BEFORE ``fully_shard`` so FSDP shards the stacked
    params. The 80B checkpoint stores per-expert weights, so the bundle remaps the
    keys at load (see ``stack_expert_state_dict``).
  * ``patch_grouped_moe_forward`` — points ``HunyuanMoE.forward`` at the grouped-GEMM
    path + pins ``BLOCK_K`` (M-invariance). Run AFTER ``fully_shard``.

The router gate (fp32, routing-sensitive) and the shared MLP keep the normal
``F.linear`` path; the SP adapter routes the gate through pad-to-M=L and the shared
``down_proj`` (K=moe_inter) through the M-invariant kernel.

Weight order: HI3 ``gate_and_up_proj`` is ``[up; gate]`` (``chunk`` -> x1=up, x2=gate;
computes ``x1 * silu(x2)``). VeOmni ``MergedFc1`` wants ``[gate; up]`` (it does
``silu(fc1_1) * fc1_2``). We swap halves once at restructure/remap, and
``silu(gate)*up == up*silu(gate)``. VeOmni applies the router weight before ``fc2``,
which equals HI3's after-``down`` weighting by linearity of ``down``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_HI3_MOE_CLASS = "HunyuanMoE"


def pin_group_gemm_block_k() -> None:
    """Force VeOmni's grouped-GEMM to one fixed config (CATCH_ALL fallback, BLOCK_K=32)
    for all ``total_M``, so the K-reduction order is M-invariant. The ``pretuned``
    algo-key buckets on ``total_M//5000``; under SP the shard's ``total_M`` lands in a
    different bucket than non-SP and could pick a different ``BLOCK_K`` — exactly the
    M-dependence we are removing. Idempotent."""
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


def stack_expert_weights(gate_and_up: list[torch.Tensor], down: list[torch.Tensor]):
    """Stack per-expert ``gate_and_up_proj`` [2I,H] (HI3 ``[up; gate]``) and ``down_proj``
    [H,I] into ``moe_gate_up`` [E,2I,H] (swapped to ``[gate; up]``) and ``moe_down``
    [E,H,I]. Shared by the meta-restructure (data copy) and the checkpoint remap."""
    E = len(gate_and_up)
    twoI, H = gate_and_up[0].shape
    I = twoI // 2
    gu = torch.empty(E, twoI, H, dtype=gate_and_up[0].dtype, device=gate_and_up[0].device)
    dn = torch.empty(E, *down[0].shape, dtype=down[0].dtype, device=down[0].device)
    for i in range(E):
        w = gate_and_up[i]
        gu[i, :I].copy_(w[I:])   # gate  (HI3 second half)
        gu[i, I:].copy_(w[:I])   # up    (HI3 first half)
        dn[i].copy_(down[i])
    return gu, dn


def restructure_hi3_experts(model: nn.Module) -> int:
    """Replace every ``HunyuanMoE.experts`` ModuleList with stacked frozen params
    ``moe_gate_up`` [E,2I,H] (``[gate; up]``) + ``moe_down`` [E,H,I]. Works on meta
    (shapes only, data filled by the checkpoint remap) or materialized (copies +
    swaps data). Frees the per-expert modules. Idempotent. Returns #blocks changed."""
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
        twoI, H = w_gu.shape
        I = twoI // 2
        rg = w_gu.requires_grad
        if w_gu.is_meta:
            gu = torch.empty(E, twoI, H, device="meta", dtype=w_gu.dtype)
            dn = torch.empty(E, *w_dn.shape, device="meta", dtype=w_dn.dtype)
        else:
            with torch.no_grad():
                gu, dn = stack_expert_weights(
                    [e.gate_and_up_proj.weight for e in experts],
                    [e.down_proj.weight for e in experts],
                )
        moe.register_parameter("moe_gate_up", nn.Parameter(gu, requires_grad=rg))
        moe.register_parameter("moe_down", nn.Parameter(dn, requires_grad=rg))
        del moe.experts
        moe._grouped = True            # type: ignore[attr-defined]
        moe._n_experts = E             # type: ignore[attr-defined]
        moe._moe_I = I                 # type: ignore[attr-defined]
        n += 1
    return n


def _grouped_moe_forward(self, hidden_states):  # noqa: ANN001
    """Patched ``HunyuanMoE.forward``: experts via veomni grouped-GEMM, router gate +
    shared MLP unchanged (their M-invariance is handled by the SP adapter's F.linear
    routing). Mirrors HI3's silu-gated, top-k-weighted MoE math."""
    from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward

    bsz, seq, hid = hidden_states.shape
    shared = self.shared_mlp(hidden_states) if self.config.use_mixed_mlp_moe else None
    with torch.autocast("cuda", enabled=False):
        topk_weights, topk_idx = self.gate(hidden_states, topk_impl="easy")
    topk_weights = topk_weights.to(hidden_states.dtype)
    combined = group_gemm_fused_moe_forward(
        num_experts=self._n_experts,
        routing_weights=topk_weights,
        selected_experts=topk_idx,
        hidden_states=hidden_states.reshape(-1, hid),
        fc1_1_weight=None,
        fc1_2_weight=None,
        fc2_weight=self.moe_down,
        fc1_1_2_weight=self.moe_gate_up,   # merged [gate; up]
    ).reshape(bsz, seq, hid)
    return shared + combined if shared is not None else combined


def patch_grouped_moe_forward(model: nn.Module) -> None:
    """Point ``HunyuanMoE.forward`` at the grouped-GEMM path + pin ``BLOCK_K``. Run
    AFTER ``fully_shard`` (the restructure must already have run on meta). Idempotent."""
    pin_group_gemm_block_k()
    for m in model.modules():
        cls = type(m)
        if cls.__name__ == _HI3_MOE_CLASS and not getattr(cls, "_unirl_grouped", False):
            cls.forward = _grouped_moe_forward
            cls._unirl_grouped = True   # type: ignore[attr-defined]
            break


__all__ = [
    "restructure_hi3_experts",
    "patch_grouped_moe_forward",
    "pin_group_gemm_block_k",
    "stack_expert_weights",
]
