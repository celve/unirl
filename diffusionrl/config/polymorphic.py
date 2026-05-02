"""Polymorphic list expansion for heterogeneous structured-config elements.

Hydra/OmegaConf does not natively type-check polymorphic list elements: a
``List[Base]`` field is validated against ``Base``, not against per-element
subclasses. This module adds a thin pre-validation pass that lets recipe
authors write::

    components:
      - name: pickscore
        weight: 1.0
        processor_id: ...
      - name: ocr
        weight: 1.0
        lang: en

…where ``name:`` resolves to a Spec dataclass registered under a Hydra
``ConfigStore`` group via ``@register_config(group="reward/component", name="<n>", target=...)``.

Three pieces:

- :func:`polymorphic_field` — declares a dataclass field as polymorphic by
  attaching ``{"group": ..., "discriminator": ...}`` to ``field.metadata``.
  No marker on the element base class — information lives on the field.
- :func:`expand_polymorphic_lists` — pre-pass that walks a ``DictConfig``,
  finds polymorphic fields via standard ``dataclasses.fields()`` introspection,
  resolves each list element's discriminator against ``ConfigStore.instance()``,
  and merges user fields onto the registered schema. Idempotent.
- :func:`polymorphic_metadata` — read accessor for the field metadata.

Run :func:`expand_polymorphic_lists` once at the ``@hydra.main`` entry, after
``OmegaConf.resolve(cfg)`` and before ``freeze(cfg)`` / ``validate(cfg)``.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import field as _dc_field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

_METADATA_KEY = "polymorphic_group"


def polymorphic_field(
    *,
    group: str,
    discriminator: str = "name",
    **field_kwargs: Any,
) -> Any:
    """Declare a dataclass field as a polymorphic list dispatched by a
    discriminator (default ``name:``) against a Hydra ConfigStore ``group``.

    Two equivalent source forms — pick whichever reads better::

        components: Tuple[BaseRewardComponentSpec, ...] = polymorphic_field(
            group="reward/component", default_factory=tuple,
        )

        items: List[BaseRewardComponentSpec] = polymorphic_field(
            group="reward/component", default_factory=list,
        )

    The source annotation is what static type checkers and IDEs see — they
    read the AST, not runtime ``__annotations__``. ``register_config`` calls
    :func:`erase_polymorphic_annotations` to rewrite the runtime annotation
    to the matching ``Tuple[Any, ...]`` / ``List[Any]`` form so OmegaConf
    accepts raw dict assignment at YAML compose time. After
    :func:`expand_polymorphic_lists` runs and ``OmegaConf.to_object``
    materializes the cfg, each list element is a concrete subclass instance.

    The metadata is read by :func:`expand_polymorphic_lists` via standard
    ``dataclasses.fields()`` introspection. No global registry; the pass owns
    no state — it queries Hydra's ConfigStore for each ``(group, name)`` pair.
    """
    metadata = dict(field_kwargs.pop("metadata", {}))
    metadata[_METADATA_KEY] = {"group": group, "discriminator": discriminator}
    return _dc_field(metadata=metadata, **field_kwargs)


def erase_polymorphic_annotations(cls: type) -> None:
    """Rewrite polymorphic_field annotations from ``Tuple[Base, ...]`` /
    ``List[Base]`` to the matching ``...[Any, ...]`` form, in place.

    OmegaConf rejects raw dict assignment to a structured ``Tuple[Base, ...]``
    field at YAML compose time (the dict isn't a ``Base`` instance). Source
    annotations remain visible to static type checkers (which read AST, not
    runtime ``__annotations__``); only the runtime view is rewritten. Called
    from ``register_config`` before dataclass promotion. Idempotent.

    Handles both pre-promoted and not-yet-promoted classes:
    - Not yet a dataclass: rewrite ``cls.__annotations__`` so the impending
      ``dataclass(cls)`` call builds Field records with the rewritten types.
    - Already a dataclass: rewrite ``cls.__annotations__`` AND each
      :class:`dataclasses.Field`'s ``.type`` attribute (OmegaConf reads both).

    Raises ``TypeError`` if a polymorphic_field is declared on a non-list and
    non-tuple field — ``polymorphic_field`` only makes sense on list-shaped
    fields, and silently coercing the annotation would mask user errors.
    """
    annotations = getattr(cls, "__annotations__", None) or {}

    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if polymorphic_metadata(f) is not None:
                erased = _erase_to_any(annotations.get(f.name, f.type), cls, f.name)
                annotations[f.name] = erased
                f.type = erased
        return

    for name in list(annotations):
        attr = cls.__dict__.get(name)
        if isinstance(attr, dataclasses.Field) and polymorphic_metadata(attr) is not None:
            annotations[name] = _erase_to_any(annotations[name], cls, name)


def _erase_to_any(annotation: Any, cls: type, field_name: str) -> Any:
    """Map ``Tuple[X, ...]`` → ``Tuple[Any, ...]`` and ``List[X]`` → ``List[Any]``.

    Handles both eager type objects and the string forward references produced
    by ``from __future__ import annotations``. Idempotent: rewriting an
    already-erased annotation is a no-op.

    Raises ``TypeError`` on anything that isn't a list or tuple. ``cls`` and
    ``field_name`` are used only for error messages.
    """
    origin = typing.get_origin(annotation)
    if origin is list:
        return List[Any]
    if origin is tuple:
        return Tuple[Any, ...]
    if isinstance(annotation, str):
        # ``from __future__ import annotations`` defers evaluation; the
        # annotation arrives here as a string. Pattern-match the prefix.
        stripped = annotation.lstrip()
        if stripped.startswith(("List[", "typing.List[", "list[")):
            return List[Any]
        if stripped.startswith(("Tuple[", "typing.Tuple[", "tuple[")):
            return Tuple[Any, ...]
    raise TypeError(
        f"{cls.__qualname__}.{field_name}: polymorphic_field requires a "
        f"Tuple[Base, ...] or List[Base] annotation; got {annotation!r}"
    )


def polymorphic_metadata(f: dataclasses.Field) -> Optional[Dict[str, str]]:
    """Return the polymorphic descriptor on ``f`` (``{"group", "discriminator"}``)
    or ``None`` if ``f`` is not a polymorphic field."""
    meta = f.metadata.get(_METADATA_KEY)
    return dict(meta) if meta is not None else None


def expand_polymorphic_lists(cfg: DictConfig) -> None:
    """Rewrite ``<discriminator>: <name>`` discriminators on polymorphic fields
    into typed structured-config elements, in place.

    Idempotent: elements that are already typed (their schema is a registered
    dataclass and they carry no discriminator) pass through untouched.

    Run after ``OmegaConf.resolve(cfg)`` and before ``freeze(cfg)``.
    """
    # Materialize the iterator before mutating (we'll replace list nodes).
    targets = list(_find_polymorphic_fields(cfg))
    for parent, fname, meta in targets:
        list_node = parent[fname]
        if list_node is None:
            continue
        group = meta["group"]
        discriminator = meta["discriminator"]
        new_elements = []
        for raw in list_node:
            new_elements.append(_expand_element(raw, group, discriminator, parent, fname))
        parent[fname] = OmegaConf.create(new_elements)


def _expand_element(
    raw: Any,
    group: str,
    discriminator: str,
    parent: DictConfig,
    fname: str,
) -> Any:
    """Resolve one list element to a typed structured node."""
    if raw is None:
        return raw
    if not OmegaConf.is_dict(raw):
        raise ValueError(f"polymorphic list at '{_full_key(parent, fname)}': element must be a mapping; got {raw!r}")
    elem = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(elem, dict):
        raise ValueError(f"polymorphic list at '{_full_key(parent, fname)}': element must be a mapping; got {elem!r}")
    type_name = elem.pop(discriminator, None)
    if type_name is None:
        # Idempotency: an already-expanded element has no discriminator but
        # carries a registered schema. Pass through untouched.
        if dataclasses.is_dataclass(OmegaConf.get_type(raw) or ()):
            return raw
        raise ValueError(
            f"polymorphic list at '{_full_key(parent, fname)}': element missing "
            f"{discriminator!r} (group={group!r}); got {raw}"
        )
    spec_cls = _load_registered_class(group, str(type_name))
    schema = OmegaConf.structured(spec_cls)
    try:
        return OmegaConf.merge(schema, elem)
    except Exception as exc:
        raise ValueError(
            f"polymorphic list at '{_full_key(parent, fname)}': "
            f"failed to merge element {discriminator}={type_name!r} "
            f"into schema {spec_cls.__name__}: {exc}"
        ) from exc


def _find_polymorphic_fields(
    node: Any,
) -> Iterator[Tuple[DictConfig, str, Dict[str, str]]]:
    """Yield (parent_node, field_name, metadata) for every polymorphic list
    field reachable from ``node`` via standard dataclass field introspection."""
    if OmegaConf.is_dict(node):
        cls = OmegaConf.get_type(node)
        if cls is not None and dataclasses.is_dataclass(cls):
            for f in dataclasses.fields(cls):
                meta = polymorphic_metadata(f)
                if meta is not None and f.name in node:
                    yield node, f.name, meta
        for key in node:
            child = node._get_node(key) if hasattr(node, "_get_node") else node[key]
            yield from _find_polymorphic_fields(child)
    elif OmegaConf.is_list(node):
        for i in range(len(node)):
            child = node._get_node(i) if hasattr(node, "_get_node") else node[i]
            yield from _find_polymorphic_fields(child)


def _load_registered_class(group: str, name: str) -> Type:
    """Look up the schema class registered at ``(group, name)`` in Hydra's
    ConfigStore. Raises ``ValueError`` with the list of known names on miss.

    ``ConfigStore.store`` converts the registered class to a structured
    ``DictConfig`` internally; we recover the underlying dataclass class via
    ``OmegaConf.get_type``.
    """
    cs = ConfigStore.instance()
    try:
        result = cs.load(f"{group}/{name}.yaml")
    except Exception as exc:
        # Hydra raises ConfigLoadError on miss; ConfigStore internals can
        # surface KeyError or AssertionError on malformed paths. Treat any
        # lookup failure as an unknown name.
        known = sorted(_registered_names_for_group(group))
        raise ValueError(
            f"unknown {name!r} for polymorphic group {group!r}; known names: {known if known else '(none registered)'}"
        ) from exc
    node = result.node
    if isinstance(node, type) and dataclasses.is_dataclass(node):
        return node
    cls = OmegaConf.get_type(node) if OmegaConf.is_config(node) else None
    if cls is not None and dataclasses.is_dataclass(cls):
        return cls
    if dataclasses.is_dataclass(node):
        return type(node)
    raise TypeError(f"registered schema at {group!r}/{name!r} is not a dataclass; got {node!r}")


def _registered_names_for_group(group: str) -> list[str]:
    """Best-effort enumeration of registered names under ``group``. Returns []
    if the group's structure can't be traversed."""
    cs = ConfigStore.instance()
    bucket: Any = cs.repo
    for part in group.split("/"):
        if not isinstance(bucket, dict) or part not in bucket:
            return []
        bucket = bucket[part]
    if not isinstance(bucket, dict):
        return []
    return [n.removesuffix(".yaml") for n in bucket if isinstance(n, str) and n.endswith(".yaml")]


def _full_key(parent: DictConfig, fname: str) -> str:
    try:
        key = parent._get_full_key(fname)
    except Exception:
        key = None
    return key or f"<root>.{fname}"


__all__ = [
    "erase_polymorphic_annotations",
    "expand_polymorphic_lists",
    "polymorphic_field",
    "polymorphic_metadata",
]
