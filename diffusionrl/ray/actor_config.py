"""Ambient cfg access for Ray actor processes.

Each Ray actor is a Python process of its own. ``ConfigActor`` stashes the
composed Hydra cfg into a module-level handle at actor construction time,
so any code running inside that process can reach the cfg via ``current()``
without receiving it as a kwarg.

Outside a Ray actor — on the driver, in tests that do not construct a
ConfigActor — ``current()`` raises. This is intentional: the ambient view
is a read on actor-process-local state, not a global app config.

The module deliberately uses a plain module-level handle rather than a
``ContextVar``. Ray actor methods run single-context, pure-sync in this
codebase; scoping primitives buy nothing here and would obscure the
"install once, read many times" contract.
"""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig

_current: Optional[DictConfig] = None


def current() -> DictConfig:
    """Return the cfg installed by the actor that owns this process."""
    if _current is None:
        raise RuntimeError("actor_config.current() called before a ConfigActor installed a cfg")
    return _current


class ConfigActor:
    """Cooperative-super base that installs ``cfg`` into the module handle.

    Place first in the MRO so ``super().__init__(cfg=..., **rest)`` from the
    subclass installs the cfg before any other parent's ``__init__`` runs.
    """

    def __init__(self, *, cfg: Optional[DictConfig] = None, **kwargs: Any) -> None:
        global _current
        if cfg is not None:
            _current = cfg
            self._cfg = cfg
        super().__init__(**kwargs)


__all__ = ["ConfigActor", "current"]
