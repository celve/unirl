"""Fail-closed compatibility shims installed on the external hymm stack; see README.md."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

_FA3_MODULE = "hymm.models.basic.flash_attn_no_pad"
_FA3_MARKER = "_unirl_fa3_varlen_compat"
# hymm passes these three by keyword and the rest positionally, so the shim's contract is
# that the cu_seqlens pair is followed by the max_seqlen pair, optionally via seqused.
_FA3_REQUIRED = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
_FA3_KEYWORDS = ("softmax_scale", "causal", "deterministic")


def _install_fa3_varlen() -> bool:
    """Adapt hymm's positional FlashAttention-3 varlen call to the installed signature."""
    module = importlib.import_module(_FA3_MODULE)
    backend = getattr(module, "flash_attn_varlen_func_v3", None)
    if backend is None:
        return False  # hymm raises ImportError itself on the flash3 path
    if getattr(backend, _FA3_MARKER, False):
        return True

    params = tuple(inspect.signature(backend).parameters)
    missing = [name for name in _FA3_REQUIRED + _FA3_KEYWORDS if name not in params]
    if missing:
        raise RuntimeError(
            f"leo2: FlashAttention-3 varlen signature {params} lacks {missing} -- the shim in "
            f"{_FA3_MODULE} cannot be applied safely. See the package README."
        )
    needs_seqused = "seqused_q" in params
    order = [params.index(name) for name in _FA3_REQUIRED]
    if order != sorted(order) or (
        needs_seqused and not params.index("cu_seqlens_k") < params.index("seqused_q") < params.index("max_seqlen_q")
    ):
        raise RuntimeError(
            f"leo2: FlashAttention-3 varlen parameters are reordered ({params}); the positional call in "
            f"{_FA3_MODULE} would pass the wrong arguments. See the package README."
        )

    def _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if needs_seqused:
            out = backend(q, k, v, cu_seqlens_q, cu_seqlens_k, None, None, max_seqlen_q, max_seqlen_k, **kwargs)
        else:
            out = backend(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    setattr(_varlen, _FA3_MARKER, True)
    module.flash_attn_varlen_func_v3 = _varlen
    logger.info("leo2: FA3 varlen shim installed (seqused=%s)", needs_seqused)
    return True


def install_hymm_compat() -> Tuple[str, ...]:
    """Install every hymm shim this package needs; idempotent, returns what was applied."""
    applied = []
    if _install_fa3_varlen():
        applied.append("fa3_varlen")
    return tuple(applied)


__all__ = ["install_hymm_compat"]
