"""SGLang sampler/runtime exports with lazy loading."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "SGLangClient": ("diffusionrl.samplers.sglang.client", "SGLangClient"),
    "SGLangClientError": ("diffusionrl.samplers.sglang.client", "SGLangClientError"),
    "SGLangProtocolError": ("diffusionrl.samplers.sglang.client", "SGLangProtocolError"),
    "SGLangTimeoutError": ("diffusionrl.samplers.sglang.client", "SGLangTimeoutError"),
    "SGLangRolloutEngine": ("diffusionrl.samplers.sglang.engine", "SGLangRolloutEngine"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
