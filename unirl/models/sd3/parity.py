"""Engine-parity attention processor for the diffusers SD3 transformer.

Installed by ``SD3Bundle`` when ``shared_kernels: true``. Makes the trainer's
attention computation **bitwise-identical** to vLLM-Omni v0.20.0's
``SD3CrossAttention.forward`` (``vllm_omni/diffusion/models/sd3/sd3_transformer.py``)
by mirroring, exactly:

1. the fused QKV GEMM (one ``[D -> 3D]`` ``F.linear`` with catted weights/bias,
   split via ``chunk(3, dim=-1)``) — NOT diffusers' three separate GEMMs;
2. the joint-attention concat order ``[text, image]`` (diffusers uses
   ``[image, text]``; both are mathematically equal, but flash-attention
   accumulates over keys in memory order, so bit-parity requires the engine's);
3. the shared attention kernel (``unirl.kernels.sd3.shared_attention`` — same
   resolved flash-attn entry, same flags) and the engine epilogue
   (``flatten(2, 3)``, ``.to(query.dtype)``, text-first split);
4. the shared qk-norm (``shared_rms_norm``, also patched into the engine);
5. the engine's LoRA application (fused base GEMM + per-slice
   ``(x_flat @ A.t()) @ B.t()`` with alpha/r folded into B beforehand —
   vllm-omni ``diffusion/lora/layers/base_linear.py::apply`` at v0.20.0),
   reading the PEFT submodules of the trainer's live adapters.

Everything outside attention (AdaLN, PatchEmbed, timestep embed, GELU-tanh FFN,
LayerNorms, residuals) is already the same diffusers module code on both sides
and needs no mirroring — provided the recipe disables autocast
(``autocast_precision: fp32``) so the trainer runs pure bf16 like the engine.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.kernels.sd3 import shared_attention, shared_rms_norm

LoraParams = Tuple[torch.Tensor, torch.Tensor, float]  # (A [r,in], B [out,r], scaling)


def _linear_params(mod: torch.nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[LoraParams]]:
    """Weights of a projection, PEFT-aware.

    For a plain ``nn.Linear`` returns ``(weight, bias, None)``. For a PEFT
    ``lora.Linear`` returns the BASE weights plus the active-adapter LoRA
    triple, so the caller can replicate the engine's apply expression instead
    of PEFT's (`base + B(A(x))*s` with scale-after differs bitwise from the
    engine's scale-folded form for non-power-of-two scales).
    """
    if hasattr(mod, "base_layer"):  # peft.tuners.lora.Linear
        base = mod.base_layer
        lora: Optional[LoraParams] = None
        lora_a = getattr(mod, "lora_A", None)
        if lora_a is not None and len(lora_a) > 0:
            adapters = list(getattr(mod, "active_adapters", None) or lora_a.keys())
            if len(adapters) != 1:
                raise RuntimeError(f"parity processor supports exactly one active adapter, got {adapters!r}")
            name = adapters[0]
            lora = (
                mod.lora_A[name].weight,
                mod.lora_B[name].weight,
                float(mod.scaling[name]),
            )
            dropout = getattr(mod, "lora_dropout", None)
            if dropout is not None and name in dropout and getattr(dropout[name], "p", 0.0) not in (0.0, None):
                raise RuntimeError("parity processor requires lora_dropout == 0 (engine applies none)")
        return base.weight, base.bias, lora
    return mod.weight, mod.bias, None


def _fused_base(mods: List[torch.nn.Module], params) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The catted base weight/bias for a multi-projection fused GEMM.

    ``torch.cat`` on weights is exact (pure data movement) but allocates a
    COPY — and under gradients F.linear saves its weight operand in the
    graph, so a per-call cat pins one full fused copy per attention per
    step-graph (wan22's serial multi-SDE-step replay held ~67 GB of these).
    When the base projections are FROZEN (LoRA training), the fused copy is
    cached on the lead module at the first grad-mode call and reused —
    autograd then saves the SAME tensor every call (a reference, not a
    copy). Lazy on grad-mode keeps never-replayed experts (wan22 low-noise)
    and no-grad generate paths cache-free. Trainable-base recipes (SD3
    full-weight parity) keep the per-call cat: gradients must flow through
    it to the per-projection params, and at SD3 scale the per-graph copies
    are small.
    """
    lead = mods[0]
    cached = getattr(lead, "_unirl_fused_base", None)
    if cached is not None:
        return cached
    weight = torch.cat([w for w, _, _ in params], dim=0)
    biases = [b for _, b, _ in params]
    if any(b is None for b in biases) and any(b is not None for b in biases):
        raise RuntimeError("parity processor: mixed bias/no-bias projections cannot be fused")
    bias = torch.cat(biases, dim=0) if biases[0] is not None else None
    if torch.is_grad_enabled() and not any(w.requires_grad for w, _, _ in params):
        lead._unirl_fused_base = (weight, bias)
    return weight, bias


def _fused_linear(mods: List[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    """One fused GEMM over the catted projection weights + engine-style LoRA.

    Mirrors the engine where q/k/v live in ONE ``QKVParallelLinear``: same GEMM
    shape (``[*, in] @ [in, sum(out)]``), bias applied in-kernel, then LoRA
    deltas added per output slice on the flattened tokens. Single-projection
    calls skip the cat entirely (a one-tensor cat is a full copy that autograd
    would save per call; the direct param is the same values — same GEMM,
    same bits).
    """
    params = [_linear_params(m) for m in mods]
    if len(mods) == 1:
        weight, bias, _ = params[0]
    else:
        weight, bias = _fused_base(mods, params)
    y = F.linear(x, weight, bias)

    if all(lora is None for _, _, lora in params):
        return y

    # Engine LoRA apply (base_linear.py::apply): x flattened to 2-D tokens,
    # delta = (x @ A.T) @ B.T per slice, added onto that slice of the base
    # output. alpha/r is folded into B up front (lora.optimize()); at bf16 a
    # power-of-two scale folds exactly, so per-forward folding here is
    # bit-identical to the engine's fold-once-at-activation.
    x_flat = x.reshape(-1, x.shape[-1])
    outs: List[torch.Tensor] = []
    offset = 0
    for (w, _, lora) in params:
        size = w.shape[0]
        y_slice = y[..., offset : offset + size]
        if lora is not None:
            A, B, scaling = lora
            B_folded = B * scaling if scaling != 1.0 else B
            delta = (x_flat @ A.t()) @ B_folded.t()
            y_slice = y_slice + delta.view(*y.shape[:-1], size)
        outs.append(y_slice)
        offset += size
    return torch.cat(outs, dim=-1)


def _plain_linear(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Single (non-fused) projection with engine-style LoRA — for to_out/to_add_out."""
    return _fused_linear([mod], x)


def _qk_norm(norm: Optional[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    if norm is None or not hasattr(norm, "weight"):
        return x  # Identity / no qk-norm
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", 1e-6)
    return shared_rms_norm(x, norm.weight, float(eps))


class SharedKernelJointAttnProcessor:
    """diffusers attention processor executing the ENGINE's attention math.

    Handles the three SD3.5 attention shapes: joint (image+text), dual/plain
    self-attention (``attn2``, ``encoder_hidden_states=None``), and the
    ``context_pre_only`` last block (context returned unprojected — the engine
    sets ``to_add_out=None`` there).
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if attention_mask is not None:
            raise RuntimeError("parity processor: SD3 path never passes an attention mask (engine uses None)")

        heads = attn.heads

        # Image stream — fused QKV, engine layout [q|k|v] then chunk.
        qkv = _fused_linear([attn.to_q, attn.to_k, attn.to_v], hidden_states)
        img_query, img_key, img_value = qkv.chunk(3, dim=-1)
        img_query = img_query.unflatten(-1, (heads, -1))
        img_key = img_key.unflatten(-1, (heads, -1))
        img_value = img_value.unflatten(-1, (heads, -1))
        img_query = _qk_norm(getattr(attn, "norm_q", None), img_query)
        img_key = _qk_norm(getattr(attn, "norm_k", None), img_key)

        if encoder_hidden_states is not None:
            qkv_add = _fused_linear(
                [attn.add_q_proj, attn.add_k_proj, attn.add_v_proj], encoder_hidden_states
            )
            txt_query, txt_key, txt_value = qkv_add.chunk(3, dim=-1)
            txt_query = txt_query.unflatten(-1, (heads, -1))
            txt_key = txt_key.unflatten(-1, (heads, -1))
            txt_value = txt_value.unflatten(-1, (heads, -1))
            txt_query = _qk_norm(getattr(attn, "norm_added_q", None), txt_query)
            txt_key = _qk_norm(getattr(attn, "norm_added_k", None), txt_key)

            # ENGINE concat order: [text, image] (reversed vs stock diffusers).
            query = torch.cat([txt_query, img_query], dim=1)
            key = torch.cat([txt_key, img_key], dim=1)
            value = torch.cat([txt_value, img_value], dim=1)
        else:
            query, key, value = img_query, img_key, img_value

        head_dim = query.shape[-1]
        out = shared_attention(
            query, key, value, softmax_scale=1.0 / (head_dim**0.5), causal=False
        )
        # Engine epilogue.
        out = out.flatten(2, 3)
        out = out.to(query.dtype)

        if encoder_hidden_states is not None:
            context_seqlen = encoder_hidden_states.shape[1]
            hidden_out = out[:, context_seqlen:, :]
            context_out = out[:, :context_seqlen, :]
            if getattr(attn, "to_add_out", None) is not None and not attn.context_pre_only:
                context_out = _plain_linear(attn.to_add_out, context_out)
            hidden_out = _plain_linear(attn.to_out[0], hidden_out)
            # diffusers to_out[1] is Dropout(p=0.0); the engine has no dropout
            # module at all — skip it (inert either way).
            return hidden_out, context_out

        return _plain_linear(attn.to_out[0], out)


def install_shared_kernels(transformer: torch.nn.Module, pretrained_path: str) -> None:
    """Put the trainer's SD3 transformer on the shared numerics contract.

    Called by ``SD3Bundle.from_config`` (eager path, ``shared_kernels: true``)
    BEFORE FSDP wrapping. Parameter-free: sets the engine-parity attention
    processor on every attention module and restores the fp32
    checkpoint-value pos_embed buffer (the engine holds fp32(checkpoint)
    natively; diffusers cast it to bf16). State-dict keys are untouched, so
    LoRA injection, FSDP wrap, and weight sync all see the stock module tree.
    """
    import logging

    from unirl.kernels.sd3 import kernel_fingerprint, restore_fp32_pos_embed

    transformer.set_attn_processor(SharedKernelJointAttnProcessor())
    restore_fp32_pos_embed(transformer, pretrained_path)
    logging.getLogger(__name__).info(
        "SD3 shared kernels installed (trainer side): %s", kernel_fingerprint()
    )


__all__ = ["SharedKernelJointAttnProcessor", "install_shared_kernels"]
