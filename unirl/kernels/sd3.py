"""Shared SD3 forward kernels — one implementation for trainer AND rollout engine.

The rollout-vs-train engine gap exists because the vLLM-Omni worker and the
FSDP trainer run different implementations of the same DiT forward. This
module is the single source of truth for the numerics-critical ops; both
sides import it:

- trainer: ``unirl/models/sd3/parity.py`` (diffusers attention processor)
- engine:  ``unirl/rollout/engine/vllm_omni/patches/runtime.py``
  (``patch_sd3_shared_kernels`` — installs these onto the vLLM-Omni worker
  pre-model-build via ``VLLMOmniHijack``)

Design rules:
- The flash-attention entrypoints are resolved from vLLM-Omni's own
  ``utils/fa.py`` module when available, so trainer and engine bind the SAME
  python objects (same compiled extension). The inline fallback chain below
  is byte-for-byte the CUDA branch of that module (v0.20.0) for environments
  where vllm_omni is not importable.
- ``shared_attention`` reproduces vLLM-Omni ``FlashAttentionImpl.forward_cuda``
  (v0.20.0) exactly for the unmasked case: prefer dense ``flash_attn_func``,
  fall back to the varlen-dense path.
- All imports of engine packages are lazy — this module must import cleanly
  in the sglang venv (no vllm/vllm_omni installed).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Flash-attention entry resolution
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def resolve_flash_entries() -> Tuple[Optional[Callable], Optional[Callable], str]:
    """Resolve ``(flash_attn_func, flash_attn_varlen_func, source)``.

    Prefers vLLM-Omni's own resolution (``utils/fa.py``) so both processes
    bind the identical objects. ``source`` is the defining module of the
    dense entry (or the varlen entry when dense is unavailable) — used for
    fingerprinting and for the differentiability check.
    """
    try:
        from vllm_omni.diffusion.attention.backends.utils import fa as _fa

        dense, varlen = _fa.flash_attn_func, _fa.flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        dense, varlen = _resolve_flash_entries_fallback()
    entry = dense if dense is not None else varlen
    source = getattr(entry, "__module__", "none") if entry is not None else "none"
    return dense, varlen, source


def _resolve_flash_entries_fallback() -> Tuple[Optional[Callable], Optional[Callable]]:
    """CUDA fallback chain — mirrors vllm-omni v0.20.0 ``utils/fa.py`` exactly."""
    dense = varlen = None
    try:
        from fa3_fwd_interface import flash_attn_func as dense  # type: ignore
        from fa3_fwd_interface import flash_attn_varlen_func as varlen  # type: ignore
    except (ImportError, ModuleNotFoundError):
        pass
    if dense is None:
        try:
            from flash_attn_interface import flash_attn_func as dense  # type: ignore
            from flash_attn_interface import flash_attn_varlen_func as varlen  # type: ignore
        except (ImportError, ModuleNotFoundError):
            pass
    if dense is None:
        try:
            from flash_attn import flash_attn_func as dense  # type: ignore
            from flash_attn import flash_attn_varlen_func as varlen  # type: ignore
        except (ImportError, ModuleNotFoundError):
            pass
    if dense is None:
        try:
            from flash_attn.flash_attn_interface import flash_attn_func as dense  # type: ignore
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as varlen  # type: ignore
        except (ImportError, ModuleNotFoundError):
            pass
    if varlen is None:
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen  # type: ignore
        except (ImportError, ModuleNotFoundError):
            pass
    return dense, varlen


def parity_debug_sha(t: Optional[torch.Tensor]) -> str:
    """Short content hash for cross-process tensor comparison (debug only).

    Upcasts to fp32 first (lossless from bf16) so the hash is dtype-portable,
    then hashes raw bytes — equal hash ⇔ bitwise-equal values.
    """
    if t is None:
        return "none"
    import hashlib

    data = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha1(data).hexdigest()[:10]


def kernel_fingerprint() -> Dict[str, str]:
    """Identify the resolved attention build — stamp into logs/RolloutResp so
    a tripped parity gate is attributable to a concrete binary."""
    dense, varlen, source = resolve_flash_entries()
    return {
        "torch": torch.__version__,
        "flash_entry": source,
        "flash_dense": "yes" if dense is not None else "no",
        "flash_varlen": "yes" if varlen is not None else "no",
    }


# ---------------------------------------------------------------------------
# Shared attention
# ---------------------------------------------------------------------------


def _unwrap_flash_output(out: Any) -> torch.Tensor:
    # FA3 may return (out, lse), FA2 returns out — same unwrap as the engine.
    return out[0] if isinstance(out, tuple) else out


def _raw_shared_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Engine-exact unmasked attention: vllm-omni v0.20.0
    ``FlashAttentionImpl.forward_cuda`` with ``attn_mask=None``.

    Layout ``[B, S, H, D]`` in and out.
    """
    dense, varlen, _ = resolve_flash_entries()
    if dense is not None:
        out = dense(query, key, value, causal=causal, softmax_scale=softmax_scale)
        return _unwrap_flash_output(out)
    if varlen is None:
        raise ImportError(
            "shared_attention requires a flash-attention build (fa3-fwd, "
            "flash-attn, or vllm.vllm_flash_attn); none importable."
        )
    # Varlen-dense fallback — byte-for-byte the engine's _forward_varlen_dense.
    batch_size, q_len = query.size()[:2]
    cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
    query = query.flatten(0, 1)
    key = key.flatten(0, 1)
    value = value.flatten(0, 1)
    out = varlen(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=q_len,
        max_seqlen_k=q_len,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    out = _unwrap_flash_output(out)
    return out.reshape(batch_size, q_len, *out.shape[1:])


# Entries with a native backward (calling them under autograd "just works"):
# the flash-attn FA2 package and FA3 source builds. fa3_fwd_interface (fwd-only
# wheel) and vllm.vllm_flash_attn (inference-only build) have no backward.
def _entry_is_differentiable() -> bool:
    _, _, source = resolve_flash_entries()
    return source.split(".")[0] in ("flash_attn", "flash_attn_interface")


class _SDPARecomputeAttention(torch.autograd.Function):
    """Forward = engine-exact flash kernel (no grad); backward = SDPA recompute.

    Used only when the resolved flash entry is forward-only. The forward BITS
    are what parity requires; the backward is a numerically-close gradient
    through torch SDPA (the logp anchor itself is computed no-grad, so the
    GRPO ratio is unaffected — only the gradient path approximates).
    """

    @staticmethod
    def forward(ctx, query, key, value, softmax_scale, causal):
        ctx.save_for_backward(query, key, value)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        with torch.no_grad():
            return _raw_shared_attention(query, key, value, softmax_scale, causal)

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        with torch.enable_grad():
            # SDPA wants [B, H, S, D].
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=ctx.causal,
                scale=ctx.softmax_scale,
            ).transpose(1, 2)
        gq, gk, gv = torch.autograd.grad(out, (q, k, v), grad_out)
        return gq, gk, gv, None, None


def shared_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """The ONE attention both trainer and engine call. ``[B, S, H, D]`` layout.

    Forward bits are identical on both sides by construction (same resolved
    kernel, same flags, same shapes). Gradient support: native when the
    resolved entry has a backward (FA2/FA3 builds); otherwise an SDPA-recompute
    autograd wrapper.
    """
    needs_grad = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )
    if needs_grad and not _entry_is_differentiable():
        return _SDPARecomputeAttention.apply(query, key, value, softmax_scale, causal)
    return _raw_shared_attention(query, key, value, softmax_scale, causal)


# ---------------------------------------------------------------------------
# Shared qk-norm
# ---------------------------------------------------------------------------


def shared_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim: fp32 accumulation, fp32 weight-mul, cast back.

    Replaces BOTH the engine's fused CUDA ``vllm...RMSNorm`` kernel and the
    trainer's diffusers eager RMSNorm for the SD3 qk-norms — one expression
    tree, so the two sides round identically.
    """
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x).to(input_dtype)


# ---------------------------------------------------------------------------
# pos_embed buffer restore (seam: engine buffer is fp32(checkpoint), trainer's
# was cast to bf16 by from_pretrained)
# ---------------------------------------------------------------------------


def restore_fp32_pos_embed(transformer: torch.nn.Module, pretrained_path: str, subfolder: str = "transformer") -> None:
    """Reload ``pos_embed.pos_embed`` from the checkpoint and hold it in fp32.

    SD3.5 checkpoints DO carry the sincos table (``pos_embed_max_size`` makes
    the buffer persistent), and its values differ wholesale from what the
    current diffusers helper would recompute (different export-time
    convention — measured maxabs 2.0 on SD3.5-medium). Both sides must
    therefore use the CHECKPOINT value:

    - engine (vLLM-Omni): loads it via ``default_weight_loader`` into an
      fp32 buffer (the loader never casts buffers) → holds fp32(stored);
    - trainer (diffusers ``from_pretrained(torch_dtype=bf16)``): casts the
      buffer to bf16 → holds bf16(stored).

    Restoring fp32(stored) on the trainer reproduces the engine bits exactly.
    Raises if the key is missing — parity cannot be silently approximated.
    """
    import glob
    import os

    from safetensors import safe_open

    key = "pos_embed.pos_embed"
    folder = os.path.join(pretrained_path, subfolder)
    shards = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"restore_fp32_pos_embed: no .safetensors under {folder!r}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            if key in f.keys():
                stored = f.get_tensor(key)
                buf = transformer.pos_embed.pos_embed
                if tuple(stored.shape) != tuple(buf.shape):
                    raise RuntimeError(
                        f"restore_fp32_pos_embed: checkpoint {key} shape "
                        f"{tuple(stored.shape)} != module buffer {tuple(buf.shape)}"
                    )
                transformer.pos_embed.pos_embed = stored.to(torch.float32).to(buf.device)
                return
    raise KeyError(
        f"restore_fp32_pos_embed: {key!r} not found in {folder!r} — this SD3 "
        f"checkpoint layout is unexpected; parity install refuses to guess."
    )


__all__ = [
    "kernel_fingerprint",
    "resolve_flash_entries",
    "restore_fp32_pos_embed",
    "shared_attention",
    "shared_rms_norm",
]
