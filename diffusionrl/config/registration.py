"""ConfigStore registration helpers.

Thin typed wrappers around ``hydra.core.config_store.ConfigStore.store``.
``register_config`` works as decorator or direct call; ``register_preset``
stores a pre-configured dataclass instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, make_dataclass
from typing import Any, Callable, Optional, Type, TypeVar, Union, overload

from hydra.core.config_store import ConfigStore

T = TypeVar("T")


@overload
def register_config(cls: Type[T], /) -> Type[T]: ...
@overload
def register_config(
    cls: Type[T],
    /,
    *,
    group: Optional[str] = ...,
    name: Optional[str] = ...,
    target: Optional[str] = ...,
    expand: bool = ...,
    package: Optional[str] = ...,
    provider: Optional[str] = ...,
    mutable: bool = ...,
) -> Type[T]: ...
@overload
def register_config(
    *,
    group: Optional[str] = ...,
    name: Optional[str] = ...,
    target: Optional[str] = ...,
    expand: bool = ...,
    package: Optional[str] = ...,
    provider: Optional[str] = ...,
    mutable: bool = ...,
) -> Callable[[Type[T]], Type[T]]: ...


def register_config(
    cls: Optional[Type[T]] = None,
    /,
    *,
    group: Optional[str] = None,
    name: Optional[str] = None,
    target: Optional[str] = None,
    expand: bool = False,
    package: Optional[str] = None,
    provider: Optional[str] = None,
    mutable: bool = False,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Register a class as a Hydra ConfigStore schema.

    Auto-promotes a plain class to a dataclass if needed, so users can drop
    the ``@dataclass`` decorator at config sites.

    Supports three call patterns:

    1. Bare decorator::

           @register_config
           class RayConfig: ...

       Registered as ``name="RayConfig"`` with no group.

    2. Decorator factory::

           @register_config(group="ray")
           class RayConfig: ...

       Registered as ``group="ray"``, ``name="ray_schema"``.

    3. Direct call (equivalent to the decorator forms, plus exposes
       ``package`` / ``provider``)::

           register_config(RayConfig, group="ray", package="_global_")

    ``name`` defaults:
        - if ``name`` is explicitly passed → used verbatim
        - elif ``group`` is passed → ``f"{group}_schema"`` (lets a schema
          coexist with presets under the same group)
        - else → ``cls.__name__``

    ``target`` (optional): when provided, the decorator synthesizes a dataclass
    *subclass* that adds ``_target_: str = "<target>"`` as a real schema field
    (kw-only, so default-ordering rules are not violated). The wrapped subclass
    is what gets stored in the ConfigStore and what the decorator returns; this
    keeps the structured-config metadata intact end-to-end so
    ``OmegaConf.to_object(cfg)`` round-trips back to a typed dataclass without
    any registry lookup.

    ``expand`` selects the ``diffusionrl.config.instantiate.build`` dispatch
    mode (only meaningful with ``target``):

    - ``False`` (default) — ``build()`` calls ``target(config=<ConfigCls>, **deps)``
      after materializing the registered dataclass. Matches the project's
      ``__init__(*, config, **kwargs)`` convention.
    - ``True`` — ``build()`` delegates to ``hydra.utils.instantiate(cfg, **deps)``,
      which calls ``target(**fields, **deps)``. For third-party classes taking
      flat kwargs natively.

    Stored as the ``_hydra_expand_`` class attribute on the registered class
    (so ``build()`` reads it via ``OmegaConf.get_type(cfg)._hydra_expand_``;
    no module-level registry).

    ``mutable`` (default ``False``): opts the schema out of the post-compose
    readonly seal applied by ``diffusionrl.config.instantiate.freeze``. A node
    whose declared class has ``_hydra_mutable_ = True`` stays writable after
    compose; everything else is sealed. Stored as a class attribute so
    inheritance through MRO is automatic — subclasses pick it up unless they
    override.

    All other kwargs (``group``, ``package``, ``provider``) are forwarded to
    ``ConfigStore.store`` unchanged.
    """
    if expand and target is None:
        raise ValueError("register_config: expand=True requires a target")

    def _do_register(inner_cls: Type[T]) -> Type[T]:
        if not isinstance(inner_cls, type):
            raise TypeError(f"register_config requires a class, got {inner_cls!r}")

        # Rewrite polymorphic field annotations so OmegaConf accepts raw dict
        # assignment for fields declared as ``Tuple[Base, ...]`` / ``List[Base]``.
        # Static type checkers read source AST and continue to see the meaningful
        # element type; only the runtime view is rewritten to ``...[Any, ...]``.
        from diffusionrl.config.polymorphic import erase_polymorphic_annotations

        erase_polymorphic_annotations(inner_cls)

        if not is_dataclass(inner_cls):
            inner_cls = dataclass(inner_cls)  # type: ignore[assignment]

        resolved_name = name if name is not None else (f"{group}_schema" if group is not None else inner_cls.__name__)

        if target is not None:
            # Frozen base requires frozen subclass (Python dataclass invariant);
            # propagate the flag so e.g. `@dataclass(frozen=True)` syncs configs
            # produce a frozen wrapped subclass.
            parent_params = getattr(inner_cls, "__dataclass_params__", None)
            parent_frozen = bool(parent_params and parent_params.frozen)
            wrapped = make_dataclass(
                inner_cls.__name__,
                fields=[("_target_", str, field(default=target, kw_only=True))],
                bases=(inner_cls,),
                frozen=parent_frozen,
            )
            wrapped.__module__ = inner_cls.__module__
            wrapped.__qualname__ = inner_cls.__qualname__
            wrapped.__doc__ = inner_cls.__doc__
            wrapped._hydra_expand_ = expand
            cls_for_registry: Type = wrapped
        else:
            cls_for_registry = inner_cls

        if mutable:
            cls_for_registry._hydra_mutable_ = True

        ConfigStore.instance().store(
            name=resolved_name,
            node=cls_for_registry,
            group=group,
            package=package,
            provider=provider,
        )
        return cls_for_registry  # type: ignore[return-value]

    if cls is None:
        return _do_register
    return _do_register(cls)


def register_preset(
    name: str,
    node: Any,
    *,
    group: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Register a configured dataclass *instance* under ``group/name``.

    Arg order mirrors ``ConfigStore.store(name, node, ...)`` so the call site
    reads like the native Hydra API. ``node`` must be a dataclass instance;
    passing a class raises ``TypeError`` (use ``register_config`` for types).
    """
    if isinstance(node, type):
        raise TypeError(
            f"register_preset expects a dataclass instance, got class {node!r}. Use register_config for types."
        )
    if not is_dataclass(node):
        raise TypeError(f"register_preset expects a dataclass instance, got {node!r}")
    ConfigStore.instance().store(
        name=name,
        node=node,
        group=group,
        package=package,
        provider=provider,
    )


__all__ = ["register_config", "register_preset"]
