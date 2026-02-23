"""Plugin path validation with optional capability contracts."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

from diffusionrl.utils import load_function

PLUGIN_NAMESPACE = "diffusionrl_plugins."
logger = logging.getLogger(__name__)


def _load_plugin_spec(module_name: str) -> Dict[str, Any]:
    module = importlib.import_module(module_name)
    spec = getattr(module, "PLUGIN_SPEC", None)
    if isinstance(spec, dict):
        return spec
    return {}


def _extract_declared_capabilities(target_cls: Any, module_spec: Dict[str, Any]) -> Optional[Dict[str, bool]]:
    if callable(getattr(target_cls, "declared_capabilities", None)):
        caps = target_cls.declared_capabilities()
        if isinstance(caps, dict):
            return {k: bool(v) for k, v in caps.items()}

    for attr_name in ("PLUGIN_CAPABILITIES", "CAPABILITIES", "capabilities"):
        caps = getattr(target_cls, attr_name, None)
        if isinstance(caps, dict):
            return {k: bool(v) for k, v in caps.items()}

    spec_caps = module_spec.get("capabilities")
    if isinstance(spec_caps, dict):
        return {k: bool(v) for k, v in spec_caps.items()}

    return None


def validate_plugin_target_path(
    *,
    target_path: str,
    kind: str,
    required_capabilities: Optional[Dict[str, bool]] = None,
) -> None:
    """
    Validate that plugin dotpath is importable and matches optional contracts.

    For built-in paths (non-diffusionrl_plugins.*), this is a no-op.
    """
    if not target_path.startswith(PLUGIN_NAMESPACE):
        return

    target_cls = load_function(target_path)
    module_name = target_path.rsplit(".", 1)[0]
    spec = _load_plugin_spec(module_name)

    declared_kind = spec.get("kind")
    if declared_kind and declared_kind != kind:
        raise ValueError(
            f"Plugin kind mismatch for {target_path}: declared kind={declared_kind}, expected kind={kind}."
        )

    if not required_capabilities:
        return

    declared_caps = _extract_declared_capabilities(target_cls, spec)
    if declared_caps is None:
        logger.warning(
            "Plugin %s has no declared capabilities; skipping capability contract check.",
            target_path,
        )
        return

    missing = [
        key
        for key, needed in required_capabilities.items()
        if bool(needed) and not bool(declared_caps.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Plugin capability mismatch for {target_path}: missing {missing}. "
            f"declared_capabilities={declared_caps}, required={required_capabilities}."
        )


__all__ = ["validate_plugin_target_path", "PLUGIN_NAMESPACE"]
