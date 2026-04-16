"""Model-bundle family registry and model-type discovery helpers."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from functools import lru_cache, partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

from diffusionrl.registry import (
    derive_registry_or_dotpath,
    register_component,
    require_subclass,
)

from .base import ModelBundle

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PACKAGES: Tuple[str, ...] = (
    "diffusionrl.models",
    "diffusionrl_plugins.models",
)

MODEL_BUNDLE_COMPONENT_FAMILY = "model_bundle"

register_model = partial(
    register_component,
    component_family=MODEL_BUNDLE_COMPONENT_FAMILY,
    class_checker=require_subclass(ModelBundle),
)


def ensure_builtin_model_registration() -> None:
    """Import built-in model modules so decorator-based registration runs."""
    importlib.import_module("diffusionrl.models.flux")
    importlib.import_module("diffusionrl.models.hunyuan")
    importlib.import_module("diffusionrl.models.mochi")
    importlib.import_module("diffusionrl.models.sd3")


def resolve_model_class(identifier: str) -> Any:
    return derive_registry_or_dotpath(
        component_family=MODEL_BUNDLE_COMPONENT_FAMILY,
        identifier=identifier,
        class_checker=require_subclass(ModelBundle),
    )


def _iter_module_names(package_name: str) -> Iterable[str]:
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return

    yield package.__name__
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.walk_packages(package_paths, package.__name__ + "."):
        yield module_info.name


def _declared_model_type(model_cls: type) -> Optional[str]:
    declared = getattr(model_cls, "declared_model_type", None)
    if callable(declared):
        value = declared()
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


@lru_cache(maxsize=1)
def discover_model_bundle_paths() -> Dict[str, str]:
    """
    Discover model bundle classes from configured model packages.

    Returns:
        Mapping model_type -> model bundle class dotpath
    """
    discovered: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for package_name in DEFAULT_MODEL_PACKAGES:
        for module_name in _iter_module_names(package_name):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                logger.debug("Skipping model discovery module %s: %s", module_name, exc)
                continue

            for _, candidate in inspect.getmembers(module, inspect.isclass):
                if candidate.__module__ != module_name:
                    continue
                if not issubclass(candidate, ModelBundle) or candidate is ModelBundle:
                    continue

                model_type = _declared_model_type(candidate)
                if not model_type:
                    continue
                dotpath = f"{candidate.__module__}.{candidate.__name__}"

                existing = discovered.get(model_type)
                if existing is None:
                    discovered[model_type] = dotpath
                    continue
                if existing == dotpath:
                    continue
                duplicates.setdefault(model_type, [existing]).append(dotpath)

    if duplicates:
        pieces = [
            f"{model_type}: {sorted(set(paths))}"
            for model_type, paths in sorted(duplicates.items())
        ]
        raise ValueError("Duplicate model_type declarations detected: " + "; ".join(pieces))

    return discovered


def derive_model_bundle_path(model_type: str) -> Optional[str]:
    """Resolve model bundle dotpath by model type."""
    if not isinstance(model_type, str) or not model_type.strip():
        return None
    return discover_model_bundle_paths().get(model_type.strip().lower())


def list_model_types() -> List[str]:
    """List discovered model types."""
    return sorted(discover_model_bundle_paths().keys())


__all__ = [
    "MODEL_BUNDLE_COMPONENT_FAMILY",
    "discover_model_bundle_paths",
    "ensure_builtin_model_registration",
    "derive_model_bundle_path",
    "resolve_model_class",
    "register_model",
    "list_model_types",
    "DEFAULT_MODEL_PACKAGES",
]
