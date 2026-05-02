"""Build runtime objects and read typed configs from registered schemas.

Two dispatch modes for ``build()``, declared on ``@register_config``:

- ``expand=False`` (default): ``target(config=<dataclass_instance>, **runtime_deps)``
- ``expand=True``: ``target(**fields, **runtime_deps)`` — delegates to
  ``hydra.utils.instantiate`` for full Hydra semantics.

The dispatch flag and the mutability opt-in are stored as class attributes
(``_hydra_expand_``, ``_hydra_mutable_``) on the schema class itself, set by
``@register_config``. There are no module-level registries; type info travels
with the schema.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def build(cfg: DictConfig, **deps: Any) -> Any:
    """Construct the runtime object declared by cfg's ``_target_``.

    Pre-condition: interpolations in the enclosing config should already be
    resolved. Callers using Hydra should invoke ``OmegaConf.resolve(cfg)`` at
    the ``@hydra.main`` entry so downstream slices are self-contained.

    Raises
    ------
    ValueError
        If ``cfg`` has no ``_target_`` (schema-only or unregistered section).
    """
    if cfg is None:
        return None
    target_path = cfg.get("_target_")
    if target_path is None:
        full_key = cfg._get_full_key(None) or "<root>"
        raise ValueError(f"cfg at '{full_key}' has no _target_ (schema-only or unregistered)")

    cls = OmegaConf.get_type(cfg)
    callable_ = hydra.utils.get_method(target_path)
    if cls is not None and getattr(cls, "_hydra_expand_", False):
        # Bypass hydra.utils.instantiate so dataclass deps (e.g.
        # PlacementConfig with @property fields like total_rollout_gpus)
        # are forwarded as Python objects, not as DictConfig copies — and
        # so a registered _target_ on a dep's class doesn't re-trigger
        # instantiation through Hydra's recursion machinery.
        from dataclasses import fields as _dc_fields

        obj = materialize(cfg)
        cfg_kwargs = {f.name: getattr(obj, f.name) for f in _dc_fields(obj) if f.name != "_target_"}
        return callable_(**cfg_kwargs, **deps)

    return callable_(config=materialize(cfg), **deps)


def materialize(cfg: DictConfig) -> Any:
    """Return cfg as the typed dataclass instance declared by its schema.

    Thin wrapper over ``OmegaConf.to_object`` with a None passthrough.
    Used by bootstraps and tests that need the typed config object without
    invoking ``_target_``.
    """
    if cfg is None:
        return None
    return OmegaConf.to_object(cfg)


def validate(cfg: DictConfig) -> None:
    """Fail-fast schema check for every registered leaf in ``cfg``.

    Walks the cfg tree and forces a dataclass round-trip on each typed
    node. Adding a new ``@register_config`` section is automatically picked
    up — no list to edit here.

    Framework-owned inline sections (``ray``, ``sync``, ``logging``, etc.)
    have no schema and are skipped.

    Runs on the driver before ``Actor.remote(...)`` so a malformed cfg
    surfaces once with a driver-side traceback instead of N times inside
    Ray worker processes.
    """
    for leaf in _iter_typed_leaves(cfg):
        materialize(leaf)


def freeze(cfg: DictConfig) -> None:
    """Apply per-schema readonly seal to ``cfg`` in place.

    Walks the cfg tree top-down. Each ``DictConfig`` node is sealed
    (``set_readonly(True)``) unless its declared schema class has
    ``_hydra_mutable_ = True`` — i.e. it was registered with
    ``register_config(..., mutable=True)`` — in which case the node is
    left writable. Framework-owned inline subsections without a registered
    schema seal by default. ``ListConfig`` items are recursed into; the
    list container's own flag is left alone.

    Mutability inherits through MRO via standard Python attribute lookup,
    so subclasses of a mutable schema are themselves mutable unless they
    override the attribute.

    Run after ``OmegaConf.resolve(cfg)`` (and after Hydra has applied any
    CLI overrides).
    """
    _apply_freeze(cfg)


def _iter_typed_leaves(node: Any) -> Any:
    """Yield every DictConfig node that has a registered structured schema."""
    if not OmegaConf.is_dict(node):
        return
    cls = OmegaConf.get_type(node)
    if cls is not None and is_dataclass(cls):
        yield node
        return
    for key in node:
        yield from _iter_typed_leaves(node[key])


def _apply_freeze(node: Any) -> None:
    if OmegaConf.is_dict(node):
        if node._is_none():
            return
        cls = OmegaConf.get_type(node)
        is_mutable = cls is not None and getattr(cls, "_hydra_mutable_", False)
        OmegaConf.set_readonly(node, not is_mutable)
        for key in node:
            _apply_freeze(node._get_node(key))
    elif OmegaConf.is_list(node):
        if node._is_none():
            return
        for i in range(len(node)):
            _apply_freeze(node._get_node(i))


__all__ = ["build", "materialize", "validate", "freeze"]
