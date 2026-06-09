"""Meta-init support for bundles feeding :class:`VeOmniBackend`.

A transformer built under ``torch.device("meta")`` and later materialized by
``to_empty()`` loses every tensor that checkpoints don't carry:

* **non-persistent registered buffers** (in ``named_buffers()`` but not in
  ``state_dict()``) — e.g. diffusers ``PatchEmbed.pos_embed`` sincos tables;
* **plain tensor attributes** in a module's ``__dict__`` — e.g. Qwen-Image's
  complex rope tables (deliberately not buffers upstream).

:func:`stamp_init_state_restore` captures that init-computed state from a
throwaway CPU-built twin and stamps a deferred op (the
``unirl.train.deferred`` contract) that restores it onto the materialized
module — the backend drains it via ``apply_deferred_ops`` *after* the
post-parallelize weight load, so persistent weights and init-computed state
never clobber each other.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


def stamp_init_state_restore(meta_model: nn.Module, cpu_twin: nn.Module) -> int:
    """Capture init-computed tensors from ``cpu_twin``; stamp the restore.

    Returns the number of captured tensors. The captured values keep the
    closure alive after the caller drops the twin; restored buffers are
    copied into the materialized (device) buffers with dtype cast, plain
    attrs are re-attached as CPU tensors (forwards ``.to(device)`` them on
    use, matching upstream behavior).
    """
    persistent = set(cpu_twin.state_dict().keys())
    buffers = {
        name: buf.detach().clone()
        for name, buf in cpu_twin.named_buffers()
        if name not in persistent
    }

    attrs = {}
    cpu_modules = dict(cpu_twin.named_modules())
    for mod_name, _meta_mod in meta_model.named_modules():
        cpu_mod = cpu_modules.get(mod_name)
        if cpu_mod is None:
            continue
        for attr, value in vars(cpu_mod).items():
            if isinstance(value, torch.Tensor):
                attrs[(mod_name, attr)] = value.detach().clone()

    if not buffers and not attrs:
        return 0

    def _restore(model: nn.Module) -> None:
        modules = dict(model.named_modules())
        for fqn, value in buffers.items():
            mod_name, _, buf_name = fqn.rpartition(".")
            owner = modules[mod_name] if mod_name else model
            live = getattr(owner, buf_name)
            live.copy_(value.to(device=live.device, dtype=live.dtype))
        for (mod_name, attr), value in attrs.items():
            owner = modules.get(mod_name)
            if owner is not None:
                owner.__dict__[attr] = value
        logger.info(
            "meta-init restore: %d non-persistent buffer(s), %d plain attr(s)",
            len(buffers),
            len(attrs),
        )

    _stamp(meta_model, _restore)
    return len(buffers) + len(attrs)


def finalize_meta_init(transformer: nn.Module, *, dtype: torch.dtype) -> nn.Module:
    """Finalize a transformer just built on the meta device for the backends'
    ``load_sharded`` path (shared by every meta-init bundle):

    * dtype-cast — on meta this only sets the dtype (no storage, no data move),
      so the backend's ``to_empty`` later materializes directly in ``dtype``;
    * stamp ``init_weights`` to a no-op — VeOmni's ``parallelize`` calls it
      unconditionally after ``to_empty``; the real weights load afterwards;
    * warn about non-persistent buffers absent from the checkpoint — if the
      model relies on their init-time values they must be restored via
      :func:`stamp_init_state_restore` (see SD3's sincos ``pos_embed`` and
      Qwen-Image's rope tables).

    ``nn.Module.to`` is in place and returns ``self``; callers rebind by
    convention. Quirk fixes that must run *before* the cast (e.g. rebuilding
    rope modules whose tables stay on meta) should be applied by the caller
    before invoking this.
    """
    transformer = transformer.to(dtype)
    transformer.init_weights = lambda: None
    non_persistent = sorted(set(n for n, _ in transformer.named_buffers()) - set(transformer.state_dict()))
    if non_persistent:
        logger.warning(
            "finalize_meta_init: %d non-persistent buffer(s) absent from the "
            "checkpoint and NOT restored by the weight load: %s%s. If the model "
            "relies on their init-time values, stamp stamp_init_state_restore.",
            len(non_persistent),
            non_persistent[:8],
            " ..." if len(non_persistent) > 8 else "",
        )
    return transformer


__all__ = ["stamp_init_state_restore", "finalize_meta_init"]
