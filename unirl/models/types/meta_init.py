"""Meta-init support for bundles feeding :class:`VeOmniBackend`.

A transformer built under ``torch.device("meta")`` and later materialized by
``to_empty()`` loses every tensor that checkpoints don't carry:

* **non-persistent registered buffers** (in ``named_buffers()`` but not in
  ``state_dict()``) — e.g. diffusers ``PatchEmbed.pos_embed`` sincos tables;
* **plain tensor attributes** in a module's ``__dict__`` — e.g. Qwen-Image's
  complex rope tables (deliberately not buffers upstream).

:func:`stamp_init_state_restore` captures that init-computed state directly
from the freshly-built model — which must be built under
``accelerate.init_empty_weights(include_buffers=False)`` so its parameters
land on meta while its buffers / ``__dict__`` tensors stay real on CPU — and
stamps a deferred op (the ``unirl.train.deferred`` contract) that restores it
onto the materialized module. The backend drains it via ``apply_deferred_ops``
*after* the post-parallelize weight load, so persistent weights and
init-computed state never clobber each other.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


def stamp_init_state_restore(model: nn.Module) -> int:
    """Capture ``model``'s own init-computed tensors; stamp the restore.

    ``model`` must be built so its non-persistent buffers and plain
    ``__dict__`` tensors are **real on CPU** (parameters may live on meta) —
    i.e. under ``accelerate.init_empty_weights(include_buffers=False)``, *not*
    ``with torch.device("meta")`` (which forces buffers to meta too). Raises
    ``ValueError`` if any captured tensor is still on meta — the tell-tale of
    the wrong build context, which would otherwise restore garbage.

    Returns the number of captured tensors. The captured values keep the
    closure alive past the build; restored buffers are copied into the
    materialized (device) buffers with dtype cast, plain attrs are re-attached
    as CPU tensors (forwards ``.to(device)`` them on use, matching upstream
    behavior).
    """
    persistent = set(model.state_dict().keys())
    buffers = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if name not in persistent
    }

    attrs = {}
    for mod_name, module in model.named_modules():
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                attrs[(mod_name, attr)] = value.detach().clone()

    if not buffers and not attrs:
        return 0

    on_meta = [name for name, value in buffers.items() if value.is_meta]
    on_meta += [f"{mod_name}.{attr}" for (mod_name, attr), value in attrs.items() if value.is_meta]
    if on_meta:
        raise ValueError(
            "stamp_init_state_restore: captured init-state is on the meta device "
            "— nothing real to restore. Build the model under "
            "accelerate.init_empty_weights(include_buffers=False) (parameters on "
            "meta, buffers/attrs real on CPU), not torch.device('meta'). "
            f"Offending tensor(s): {on_meta[:8]}"
        )

    def _restore(materialized: nn.Module) -> None:
        modules = dict(materialized.named_modules())
        for fqn, value in buffers.items():
            mod_name, _, buf_name = fqn.rpartition(".")
            owner = modules[mod_name] if mod_name else materialized
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

    _stamp(model, _restore)
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
