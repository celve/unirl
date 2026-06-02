"""Polymorphic field expansion for heterogeneous structured-config elements.

Hydra/OmegaConf does not natively type-check polymorphic elements. This module
adds a thin pre-validation pass that resolves discriminator values against
Hydra's ``ConfigStore`` and merges user fields onto the registered schema.

Three supported field forms:

**Single** — one polymorphic config, selected by string name or dict with
discriminator::

    engine: BaseEngineConfig = polymorphic_field(group="rollout/engine")

    # YAML: engine: sglang           (string shorthand)
    # YAML: engine:                   (dict with override)
    #         name: sglang
    #         tp_size: 4

**Dict** — named collection of polymorphic configs, key serves as default
discriminator::

    components: Dict[str, BaseRewardComponentSpec] = polymorphic_field(
        group="reward/component", default_factory=dict,
    )

    # YAML: components:
    #         pickscore:               (key = config name, value = overrides)
    #           weight: 0.5

**List / Tuple** — ordered collection with per-element discriminator::

    items: Tuple[BaseSpec, ...] = polymorphic_field(
        group="some/group", default_factory=tuple,
    )

    # YAML: items:
    #         - name: alpha
    #           color: blue

Three pieces:

- :func:`polymorphic_field` — declares a dataclass field as polymorphic.
- :func:`expand_polymorphic_fields` — pre-pass that resolves discriminators.
- :func:`erase_polymorphic_annotations` — rewrites runtime annotations so
  OmegaConf accepts raw values at compose time.

Run :func:`expand_polymorphic_fields` once at the ``@hydra.main`` entry, after
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
    """Declare a dataclass field as polymorphic, dispatched against a Hydra
    ConfigStore ``group``.

    Supports three annotation forms:

    - **Single**: ``field: BaseSpec = polymorphic_field(group=...)``
    - **Dict**: ``field: Dict[str, BaseSpec] = polymorphic_field(group=..., default_factory=dict)``
    - **Tuple/List**: ``field: Tuple[BaseSpec, ...] = polymorphic_field(group=..., default_factory=tuple)``

    The source annotation is what static type checkers and IDEs see.
    ``register_config`` calls :func:`erase_polymorphic_annotations` to rewrite
    the runtime annotation so OmegaConf accepts raw values at compose time.
    """
    metadata = dict(field_kwargs.pop("metadata", {}))
    metadata[_METADATA_KEY] = {"group": group, "discriminator": discriminator}
    return _dc_field(metadata=metadata, **field_kwargs)


def erase_polymorphic_annotations(cls: type) -> None:
    """Rewrite polymorphic_field annotations to OmegaConf-compatible forms.

    - ``Tuple[Base, ...]`` / ``List[Base]`` → ``Tuple[Any, ...]`` / ``List[Any]``
    - ``Dict[str, Base]`` → ``Dict[str, Any]``
    - Bare ``Base`` (single form) → ``Any``

    Source annotations remain visible to static type checkers (which read AST,
    not runtime ``__annotations__``); only the runtime view is rewritten.
    Called from ``register_config`` before dataclass promotion. Idempotent.
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
    """Map typed polymorphic annotations to OmegaConf-compatible forms.

    - ``Tuple[X, ...]`` → ``Tuple[Any, ...]``
    - ``List[X]`` → ``List[Any]``
    - ``Dict[str, X]`` → ``Dict[str, Any]``
    - Bare class ``X`` → ``Any``

    Handles both eager type objects and string forward references produced
    by ``from __future__ import annotations``. Idempotent.
    """
    origin = typing.get_origin(annotation)
    if origin is list:
        return List[Any]
    if origin is tuple:
        return Tuple[Any, ...]
    if origin is dict:
        return Dict[str, Any]
    if isinstance(annotation, type):
        return Any
    if isinstance(annotation, str):
        stripped = annotation.lstrip()
        if stripped.startswith(("List[", "typing.List[", "list[")):
            return List[Any]
        if stripped.startswith(("Tuple[", "typing.Tuple[", "tuple[")):
            return Tuple[Any, ...]
        if stripped.startswith(("Dict[", "typing.Dict[", "dict[")):
            return Dict[str, Any]
        # Bare class name (single form)
        return Any
    raise TypeError(
        f"{cls.__qualname__}.{field_name}: polymorphic_field requires a "
        f"class, Dict[str, Base], Tuple[Base, ...], or List[Base] annotation; "
        f"got {annotation!r}"
    )


def polymorphic_metadata(f: dataclasses.Field) -> Optional[Dict[str, str]]:
    """Return the polymorphic descriptor on ``f`` (``{"group", "discriminator"}``)
    or ``None`` if ``f`` is not a polymorphic field."""
    meta = f.metadata.get(_METADATA_KEY)
    return dict(meta) if meta is not None else None


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def expand_polymorphic_fields(cfg: DictConfig) -> None:
    """Resolve polymorphic field values into typed structured-config nodes.

    Dispatches on the field's erased type annotation:

    - **List/Tuple**: each element resolved by its discriminator field.
    - **Dict**: each value resolved; key is the default discriminator.
    - **Single**: the field value itself is resolved (string or dict).

    Idempotent: already-typed elements pass through untouched.
    Run after ``OmegaConf.resolve(cfg)`` and before ``freeze(cfg)``.
    """
    targets = list(_find_polymorphic_fields(cfg))
    for parent, fname, meta, field_type in targets:
        node = parent[fname]
        if node is None:
            continue
        group = meta["group"]
        discriminator = meta["discriminator"]

        form = _detect_form(field_type)
        if form == "list":
            new_elements = []
            for raw in node:
                new_elements.append(_expand_list_element(raw, group, discriminator, parent, fname))
            parent[fname] = OmegaConf.create(new_elements)
        elif form == "dict":
            new_dict = {}
            for key in node:
                new_dict[key] = _expand_dict_entry(key, node[key], group, discriminator, parent, fname)
            parent[fname] = OmegaConf.create(new_dict)
        else:
            parent[fname] = _expand_single(node, group, discriminator, parent, fname)


expand_polymorphic_lists = expand_polymorphic_fields


def _detect_form(field_type: Any) -> str:
    """Determine the polymorphic form from the erased field type."""
    origin = typing.get_origin(field_type)
    if origin in (list, tuple):
        return "list"
    if origin is dict:
        return "dict"
    return "single"


# ---------------------------------------------------------------------------
# Per-form expansion helpers
# ---------------------------------------------------------------------------


def _expand_list_element(
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


def _expand_dict_entry(
    key: str,
    raw: Any,
    group: str,
    discriminator: str,
    parent: DictConfig,
    fname: str,
) -> Any:
    """Resolve one dict entry to a typed structured node.

    Resolution order for the config name:
    1. If value is a string → use the string as config name.
    2. If value is a dict with a discriminator field → use that.
    3. Otherwise → use the dict key as config name.
    """
    if raw is None:
        spec_cls = _load_registered_class(group, key)
        return OmegaConf.structured(spec_cls)

    if isinstance(raw, str):
        spec_cls = _load_registered_class(group, raw)
        return OmegaConf.structured(spec_cls)

    if OmegaConf.is_dict(raw):
        if dataclasses.is_dataclass(OmegaConf.get_type(raw) or ()):
            return raw
        elem = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(elem, dict):
            raise ValueError(
                f"polymorphic dict at '{_full_key(parent, fname)}[{key}]': value must be a mapping; got {elem!r}"
            )
        type_name = elem.pop(discriminator, None) or key
        spec_cls = _load_registered_class(group, str(type_name))
        schema = OmegaConf.structured(spec_cls)
        try:
            return OmegaConf.merge(schema, elem)
        except Exception as exc:
            raise ValueError(
                f"polymorphic dict at '{_full_key(parent, fname)}[{key}]': "
                f"failed to merge into schema {spec_cls.__name__}: {exc}"
            ) from exc

    raise ValueError(
        f"polymorphic dict at '{_full_key(parent, fname)}[{key}]': "
        f"value must be a string, mapping, or null; got {raw!r}"
    )


def _expand_single(
    raw: Any,
    group: str,
    discriminator: str,
    parent: DictConfig,
    fname: str,
) -> Any:
    """Resolve a single polymorphic field value to a typed structured node.

    Accepts a string (config name shorthand) or a dict with discriminator.
    """
    if isinstance(raw, str):
        spec_cls = _load_registered_class(group, raw)
        return OmegaConf.structured(spec_cls)

    if OmegaConf.is_dict(raw):
        if dataclasses.is_dataclass(OmegaConf.get_type(raw) or ()):
            return raw
        elem = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(elem, dict):
            raise ValueError(
                f"polymorphic field '{_full_key(parent, fname)}': value must be a string or mapping; got {elem!r}"
            )
        type_name = elem.pop(discriminator, None)
        if type_name is None:
            raise ValueError(
                f"polymorphic field '{_full_key(parent, fname)}': "
                f"dict value missing {discriminator!r} (group={group!r}); got keys {sorted(elem.keys())}"
            )
        spec_cls = _load_registered_class(group, str(type_name))
        schema = OmegaConf.structured(spec_cls)
        try:
            return OmegaConf.merge(schema, elem)
        except Exception as exc:
            raise ValueError(
                f"polymorphic field '{_full_key(parent, fname)}': "
                f"failed to merge {discriminator}={type_name!r} "
                f"into schema {spec_cls.__name__}: {exc}"
            ) from exc

    raise ValueError(f"polymorphic field '{_full_key(parent, fname)}': value must be a string or mapping; got {raw!r}")


# ---------------------------------------------------------------------------
# Field discovery
# ---------------------------------------------------------------------------


def _find_polymorphic_fields(
    node: Any,
) -> Iterator[Tuple[DictConfig, str, Dict[str, str], Any]]:
    """Yield (parent_node, field_name, metadata, field_type) for every
    polymorphic field reachable from ``node``."""
    if OmegaConf.is_dict(node):
        cls = OmegaConf.get_type(node)
        if cls is not None and dataclasses.is_dataclass(cls):
            for f in dataclasses.fields(cls):
                meta = polymorphic_metadata(f)
                if meta is not None and f.name in node:
                    yield node, f.name, meta, f.type
        for key in node:
            child = node._get_node(key) if hasattr(node, "_get_node") else node[key]
            yield from _find_polymorphic_fields(child)
    elif OmegaConf.is_list(node):
        for i in range(len(node)):
            child = node._get_node(i) if hasattr(node, "_get_node") else node[i]
            yield from _find_polymorphic_fields(child)


# ---------------------------------------------------------------------------
# ConfigStore lookup
# ---------------------------------------------------------------------------


def _load_registered_class(group: str, name: str) -> Type:
    """Look up the schema class registered at ``(group, name)`` in Hydra's
    ConfigStore. Raises ``ValueError`` with the list of known names on miss.
    """
    cs = ConfigStore.instance()
    try:
        result = cs.load(f"{group}/{name}.yaml")
    except Exception as exc:
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
    """Best-effort enumeration of registered names under ``group``."""
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
    "expand_polymorphic_fields",
    "expand_polymorphic_lists",
    "polymorphic_field",
    "polymorphic_metadata",
]
