"""Engine-parity kernels for the diffusers Wan2.2 transformer (both experts).

Installed by ``WAN22Bundle`` when ``shared_kernels: true``. Makes the
trainer's forward **bitwise-identical** to vLLM-Omni v0.20.0's
``wan2_2_transformer.py`` on the same GPU, by mirroring exactly:

1. **Attention** (``SharedKernelWanAttnProcessor``):
   - self-attention: the engine's fused QKV GEMM (one ``[D -> 3D]``
     ``F.linear`` with catted weights, split in thirds), qk-norm on the FLAT
     ``heads*head_dim`` width BEFORE the head split (``shared_rms_norm`` —
     also patched into the engine's ``vllm_omni.diffusion.layers.norm.RMSNorm``),
     the engine's RoPE application (cos/sin strided-sliced then cast to the
     activation dtype, rotate multiply in bf16 — diffusers multiplies in
     fp32), and the shared flash-attention kernel + engine epilogue;
   - cross-attention: separate q/k/v GEMMs (the engine keeps them separate),
     same flat-width qk-norm, no RoPE, no key-padding mask (both sides attend
     over the zero-padded 512 text slots);
   - the engine's LoRA expression on every projection (fused base GEMM +
     per-slice ``(x @ Aᵀ) @ B_foldedᵀ`` with alpha/r folded into B — reused
     from ``unirl.models.sd3.parity``: the fold/apply expression must have
     ONE implementation for the contract to hold).

2. **Block forward** (``_engine_order_block_forward``, bound per block
   instance): the engine computes the AdaLN modulation in **bf16**
   (``scale_shift_table + temb`` with no ``.float()``; LayerNorm is
   fp32-internal but rounds to bf16 BEFORE the ``* (1+scale) + shift``) and
   keeps residual adds in bf16 — diffusers upcasts all of these to fp32.
   Binding a reimplementation on the block instances leaves the class, the
   state dict, FSDP block wrapping, and activation checkpointing untouched.

3. **Final norm** (``_EngineOrderFinalNorm`` swapped over ``norm_out``):
   the model-level tail multiplies ``norm_out(x.float()) * (1+scale) +
   shift`` — with stock diffusers the norm returns fp32 so the modulation
   runs in fp32; the engine's AdaLayerNorm rounds the LN output to bf16
   first. Returning bf16 from the swapped module makes the (untouched)
   inline tail compute in bf16 exactly like the engine. ``norm_out`` is
   parameter-free (elementwise_affine=False), so the swap is
   state-dict-neutral.

4. **Time embedding** (``parity_time_text_embedding_forward`` +
   ``det_skinny_linear``): two seams, one function.
   (a) FSDP2's root ``fully_shard`` wrap casts floating forward inputs to
   ``param_dtype`` (``cast_forward_inputs`` defaults True), so the trainer's
   model-level ``timestep`` arrives bf16-rounded (989.5833 → 988) while the
   engine's stays fp32 — an irreversible input difference invisible outside
   FSDP (probes pass; step 0 matches because t=1000 is bf16-exact). The
   shared forward rounds t to bf16 on BOTH sides before the sinusoid.
   (b) The M=1 GEMV chain is kept off cuBLAS entirely
   (``det_skinny_linear``: explicit fp32 broadcast-mul+sum): skinny-GEMM
   algorithm selection is sensitive to operand pointer alignment (verified
   directly — a sub-16-byte offset copy of the same weights changes the
   bf16 GEMV bits), and trainer weights can live as flat-buffer views under
   sharded setups. The engine gets the same function via
   ``patch_wan22_shared_kernels``.

Already identical on both sides (verified against the v0.20.0 source — no
mirroring needed): the text-projection branch of the condition embedder
(M=512 GEMMs — fat enough that cuBLAS never re-ranks them), the fp64→fp32
RoPE frequency tables, GELU-tanh FFN, the final ``scale_shift_table + temb``
(bf16 on both), ``proj_out``, and patchify/unpatchify — provided the recipe
disables autocast (``autocast_precision: fp32``) so the trainer runs pure
bf16 like the engine.

T2V only: I2V's ``add_k_proj`` image branch and TI2V's per-token timesteps
raise rather than silently diverge.
"""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.kernels.sd3 import shared_attention, shared_rms_norm
from unirl.models.sd3.parity import _fused_linear, _plain_linear


def _rms_eps(norm: torch.nn.Module) -> float:
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", 1e-6)
    return float(eps)


def _engine_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """vllm-omni ``RotaryEmbeddingWan.forward_native`` (interleaved/GPT-J,
    half-head-dim cos/sin) — multiply happens in the activation dtype."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack(
        (
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(-2, -1).to(x.dtype)


class SharedKernelWanAttnProcessor:
    """diffusers Wan attention processor executing the ENGINE's attention math."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if attention_mask is not None:
            raise RuntimeError(
                "wan22 parity processor: no attention mask expected (engine passes "
                "none at SP=1; text is attended over its zero padding on both sides)"
            )
        if getattr(attn, "add_k_proj", None) is not None:
            raise RuntimeError("wan22 parity processor supports T2V only (I2V add_k_proj branch present)")

        heads = attn.heads

        if encoder_hidden_states is None:
            # --- self-attention: engine WanSelfAttention.forward ---
            qkv = _fused_linear([attn.to_q, attn.to_k, attn.to_v], hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            # qk-norm on the FLAT heads*head_dim width, BEFORE the head split.
            query = shared_rms_norm(query, attn.norm_q.weight, _rms_eps(attn.norm_q))
            key = shared_rms_norm(key, attn.norm_k.weight, _rms_eps(attn.norm_k))
            query = query.unflatten(2, (heads, -1))
            key = key.unflatten(2, (heads, -1))
            value = value.unflatten(2, (heads, -1))
            if rotary_emb is not None:
                # Engine slices the fp32 tables (cos even / sin odd lanes) and
                # casts to the activation dtype BEFORE the rotate multiply.
                freqs_cos, freqs_sin = rotary_emb
                cos = freqs_cos[..., 0::2].to(hidden_states.dtype)
                sin = freqs_sin[..., 1::2].to(hidden_states.dtype)
                query = _engine_rope(query, cos, sin)
                key = _engine_rope(key, cos, sin)
        else:
            # --- cross-attention: engine WanCrossAttention.forward ---
            query = _plain_linear(attn.to_q, hidden_states)
            query = shared_rms_norm(query, attn.norm_q.weight, _rms_eps(attn.norm_q))
            key = _plain_linear(attn.to_k, encoder_hidden_states)
            value = _plain_linear(attn.to_v, encoder_hidden_states)
            key = shared_rms_norm(key, attn.norm_k.weight, _rms_eps(attn.norm_k))
            query = query.unflatten(2, (heads, -1))
            key = key.unflatten(2, (heads, -1))
            value = value.unflatten(2, (heads, -1))

        head_dim = query.shape[-1]
        out = shared_attention(query, key, value, softmax_scale=1.0 / (head_dim**0.5), causal=False)
        # Engine epilogue.
        out = out.flatten(2, 3)
        out = out.to(query.dtype)
        out = _plain_linear(attn.to_out[0], out)
        # diffusers to_out[1] / the engine's self.dropout are both Dropout(0.0)
        # — inert, skipped on both sides' parity path.
        return out


def _fp32_ln(x: torch.Tensor, normalized_shape, weight, bias, eps: float) -> torch.Tensor:
    """vllm-omni ``LayerNorm.forward_native``: fp32 math, round back to the
    input dtype BEFORE any downstream modulation."""
    origin_dtype = x.dtype
    return F.layer_norm(
        x.float(),
        normalized_shape,
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(origin_dtype)


def _engine_order_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """Engine ``WanTransformerBlock.forward`` (wan2_2_transformer.py:633-673):
    bf16 modulation add, bf16 gate/residual math, fp32-internal LayerNorms
    that round to bf16 before modulation. Bound per block instance by
    ``install_shared_kernels``."""
    if temb.ndim != 3:
        raise RuntimeError(
            f"wan22 parity block forward supports the T2V [B, 6, D] modulation only "
            f"(got temb.ndim={temb.ndim}; TI2V per-token timesteps are out of scope)"
        )
    # ENGINE: no .float() on temb — the modulation params stay bf16.
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table + temb
    ).chunk(6, dim=1)

    # 1. Self-attention
    norm_hidden_states = (
        _fp32_ln(hidden_states, self.norm1.normalized_shape, self.norm1.weight, self.norm1.bias, self.norm1.eps)
        * (1 + scale_msa)
        + shift_msa
    ).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

    # 2. Cross-attention
    if isinstance(self.norm2, torch.nn.Identity):
        norm_hidden_states = hidden_states
    else:
        norm_hidden_states = _fp32_ln(
            hidden_states, self.norm2.normalized_shape, self.norm2.weight, self.norm2.bias, self.norm2.eps
        ).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        _fp32_ln(hidden_states, self.norm3.normalized_shape, self.norm3.weight, self.norm3.bias, self.norm3.eps)
        * (1 + c_scale_msa)
        + c_shift_msa
    ).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

    return hidden_states


def _engine_order_patch_embed_forward(self, x: torch.Tensor) -> torch.Tensor:
    """vllm ``Conv3dLayer._forward_mulmat`` — the engine's patchify on
    torch>=2.9: unfold the (kernel==stride, zero-padding) conv into ONE GEMM.
    diffusers runs cudnn ``F.conv3d`` — a different reduction, different bits.
    Bound onto the trainer's ``nn.Conv3d`` instance (params/state-dict
    untouched)."""
    B, C, T, H, W = x.shape
    K1, K2, K3 = self.kernel_size
    T2, H2, W2 = T // K1, H // K2, W // K3
    x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, C * K1 * K2 * K3)
    x = F.linear(x, self.weight.view(self.out_channels, -1), self.bias)
    return x.view(B, T2, H2, W2, self.out_channels).permute(0, 4, 1, 2, 3)


class _EngineOrderFinalNorm(torch.nn.Module):
    """Drop-in for the model-level ``norm_out`` (parameter-free FP32LayerNorm).

    The model tail feeds it ``hidden_states.float()`` and multiplies the
    result by the (bf16) modulation inline. Stock diffusers returns fp32
    there → fp32 modulation; the engine's AdaLayerNorm rounds the LN output
    to the model dtype first → bf16 modulation. Returning the model dtype
    reproduces the engine's rounding point without touching the model code.
    """

    def __init__(self, normalized_shape, eps: float, out_dtype: torch.dtype) -> None:
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps).to(self.out_dtype)


def _parity_ctx_state(torch_mod, os_mod):
    """One-line snapshot of every process/thread state knob that can steer
    cuBLAS/cuDNN kernel selection — printed from BOTH parity contexts at the
    divergent step to diff them."""
    import threading

    t = torch_mod
    parts = [
        ("det", t.are_deterministic_algorithms_enabled()),
        ("det_warn", t.is_deterministic_algorithms_warn_only_enabled()),
        ("cudnn_det", t.backends.cudnn.deterministic),
        ("cudnn_bench", t.backends.cudnn.benchmark),
        ("tf32", t.backends.cuda.matmul.allow_tf32),
        ("bf16red", t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        ("fp16red", t.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        ("f32prec", t.get_float32_matmul_precision()),
        ("blas", str(t.backends.cuda.preferred_blas_library())),
        ("linalg", str(t.backends.cuda.preferred_linalg_library())),
        ("inference", t.is_inference_mode_enabled()),
        ("grad", t.is_grad_enabled()),
        ("stream", t.cuda.current_stream().stream_id),
        ("dev", t.cuda.current_device()),
        ("thread", threading.current_thread().name),
        ("ws_cfg", os_mod.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        ("lt_ws", os_mod.environ.get("CUBLASLT_WORKSPACE_SIZE")),
    ]
    return " ".join(f"{k}={v}" for k, v in parts)


def det_skinny_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Deterministic cuBLAS-free linear for tiny-batch (skinny/GEMV) calls.

    cuBLAS(Lt) picks algorithms per call context — for M=1/skinny problems the
    choice (split-k vs not) changes the fp reduction ORDER, so the same bits
    through the same weights can produce different results in two processes,
    or even in one process before/after the allocator's OOM-retry clears the
    cached cuBLAS workspaces (observed in vivo: the trainer's replay flipped
    the wan2.2 time-embedder GEMV from replay step 1 onward while the fat
    block GEMMs never moved). Bit-parity therefore cannot ride on cuBLAS for
    these shapes. This computes ``x @ weight.T + bias`` as an explicit
    broadcast-multiply + ``sum`` in fp32 — torch's fixed-shape tree reduction,
    no cuBLAS, bitwise-stable across processes for a given torch build — and
    rounds once to ``x.dtype``. Chunked over output features to bound the
    fp32 intermediate (peak chunk ≈ M·8192·K·4 bytes).

    Used by ``parity_time_text_embedding_forward`` on BOTH sides of the
    contract; keep it the single implementation.
    """
    if x.shape[0] > 32:
        raise RuntimeError(f"det_skinny_linear is for tiny-batch calls, got M={x.shape[0]}")
    xf = x.float()
    outs = []
    for i in range(0, weight.shape[0], 8192):
        w = weight[i : i + 8192].float()
        acc = (xf.unsqueeze(1) * w.unsqueeze(0)).sum(-1)
        if bias is not None:
            acc = acc + bias[i : i + 8192].float()
        outs.append(acc)
    return torch.cat(outs, dim=1).to(x.dtype)


def parity_time_text_embedding_forward(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    timestep_seq_len: Optional[int] = None,
):
    """``WanTimeTextImageEmbedding.forward`` replica with a deterministic
    time path — installed on BOTH the trainer expert (MethodType bind) and
    the engine class (module patch), replacing the module code that is
    line-identical on the two sides but NOT bit-stable (see
    ``det_skinny_linear``). The text path keeps the stock module GEMMs
    ([512, 4096] — fat enough that cuBLAS never re-ranks them; probe- and
    in-vivo-verified equal). T2V only.
    """
    if timestep_seq_len is not None:
        raise RuntimeError("parity time embedding: TI2V per-token timesteps are not supported (t2v only)")
    if encoder_hidden_states_image is not None or self.image_embedder is not None:
        raise RuntimeError("parity time embedding: I2V image conditioning is not supported (t2v only)")

    te = self.time_embedder
    if getattr(te, "cond_proj", None) is not None or getattr(te, "post_act", None) is not None:
        raise RuntimeError("parity time embedding: unexpected TimestepEmbedding extras (cond_proj/post_act)")

    # Round the timestep to bf16 BEFORE the sinusoid, on both sides. The
    # trainer's t arrives already bf16-rounded — FSDP2's root ``fully_shard``
    # wrap casts every floating forward input to ``param_dtype``
    # (``MixedPrecisionPolicy.cast_forward_inputs`` defaults to True), and
    # bf16 rounding is not invertible (989.5833 → 988), so the fp32 value
    # cannot be recovered inside the model. Applying the same rounding here
    # makes the engine (which receives fp32 t) condition on the identical
    # value — the SD3 parity convention. Exact at step 0 (t=1000 is
    # bf16-representable); a ≤0.2% t quantization elsewhere, identical on
    # both sides.
    timestep = timestep.to(torch.bfloat16)

    sinusoid = self.timesteps_proj(timestep)
    w_dtype = te.linear_1.weight.dtype
    if sinusoid.dtype != w_dtype and w_dtype != torch.int8:
        sinusoid = sinusoid.to(w_dtype)
    temb = det_skinny_linear(sinusoid, te.linear_1.weight, te.linear_1.bias)
    temb = det_skinny_linear(F.silu(temb), te.linear_2.weight, te.linear_2.bias)
    temb = temb.type_as(encoder_hidden_states)
    timestep_proj = det_skinny_linear(F.silu(temb), self.time_proj.weight, self.time_proj.bias)

    import os as _os_dbg

    if _os_dbg.path.exists("/tmp/unirl_parity_debug"):
        t0 = float(timestep.reshape(-1)[0])
        if 985.0 < t0 < 995.0:
            from unirl.kernels.sd3 import parity_debug_sha as _sha_dbg

            print(
                "[wan22-tpath] grad=%s t0=%.17g t_dt=%s t_stride=%s t=%s sin=%s w1=%s b1=%s l2w=%s temb=%s enc_dt=%s"
                % (
                    torch.is_grad_enabled(),
                    t0,
                    timestep.dtype,
                    tuple(timestep.stride()),
                    _sha_dbg(timestep.contiguous()),
                    _sha_dbg(sinusoid),
                    _sha_dbg(te.linear_1.weight),
                    _sha_dbg(te.linear_1.bias),
                    _sha_dbg(te.linear_2.weight),
                    _sha_dbg(temb),
                    encoder_hidden_states.dtype,
                ),
                flush=True,
            )

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    return temb, timestep_proj, encoder_hidden_states, None


def install_shared_kernels(transformer: torch.nn.Module) -> None:
    """Put ONE trainer Wan transformer (one expert) on the shared numerics
    contract. Called by ``WAN22Bundle.from_config`` for both experts (eager
    path, ``shared_kernels: true``) BEFORE FSDP wrapping and PEFT injection
    (the processor reads PEFT submodules at call time, so injection order
    does not matter). State-dict keys are untouched.
    """
    import logging
    import os

    from unirl.kernels.sd3 import kernel_fingerprint

    if any(p.is_meta for p in transformer.parameters()):
        raise RuntimeError(
            "wan22 install_shared_kernels: transformer is meta-initialized; "
            "shared-kernel parity requires the eager load path "
            "(meta_init_transformer: false)."
        )

    # Debug-only bisect switch: install a subset of the parity surfaces
    # (comma list of processor|blocks|patchify|norm_out|time_embed). Unset = all.
    parts_env = os.environ.get("UNIRL_WAN22_PARITY_PARTS")
    parts = {p.strip() for p in parts_env.split(",") if p.strip()} if parts_env else None

    def _on(name: str) -> bool:
        return parts is None or name in parts

    if _on("processor"):
        transformer.set_attn_processor(SharedKernelWanAttnProcessor())
    if _on("blocks"):
        for block in transformer.blocks:
            block.forward = MethodType(_engine_order_block_forward, block)
    if _on("patchify"):
        pe = transformer.patch_embedding
        if (
            tuple(pe.kernel_size) != tuple(pe.stride)
            or any(pe.padding)
            or any(d != 1 for d in pe.dilation)
            or pe.groups != 1
        ):
            raise RuntimeError(
                "wan22 install_shared_kernels: patch_embedding is not a plain "
                "kernel==stride patchify conv — the engine's mulmat decomposition "
                "does not apply."
            )
        pe.forward = MethodType(_engine_order_patch_embed_forward, pe)
    if _on("time_embed"):
        ce = transformer.condition_embedder
        ce.forward = MethodType(parity_time_text_embedding_forward, ce)
    if _on("norm_out"):
        norm_out = transformer.norm_out
        if norm_out.weight is not None or norm_out.bias is not None:
            raise RuntimeError("wan22 install_shared_kernels: expected a parameter-free norm_out (affine=False)")
        transformer.norm_out = _EngineOrderFinalNorm(
            norm_out.normalized_shape, norm_out.eps, out_dtype=transformer.dtype
        )
    logging.getLogger(__name__).info(
        "wan22 shared kernels installed (trainer side, parts=%s): %s",
        parts_env or "all",
        kernel_fingerprint(),
    )


__all__ = [
    "SharedKernelWanAttnProcessor",
    "_parity_ctx_state",
    "det_skinny_linear",
    "install_shared_kernels",
    "parity_time_text_embedding_forward",
]
