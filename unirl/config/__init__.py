"""Hydra-native config surface.

Public entry points:
  - ``register_config`` / ``register_preset`` (``registration``): decorators +
    direct calls to populate Hydra's ``ConfigStore``.
  - ``build`` / ``materialize`` / ``validate`` / ``freeze`` (``instantiate``):
    construct runtime objects, return typed dataclass instances, fail-fast
    validate, or seal a composed cfg.
  - ``PrecisionName`` / ``validate_precision_type`` (``validation``): shared
    helpers used by registered dataclasses' ``__post_init__``.
  - ``require`` (``require``): one-line precondition helper for ``__post_init__``
    and cross-component validators.
"""

from __future__ import annotations

from unirl.config.instantiate import build, freeze, materialize, validate
from unirl.config.registration import register_config, register_preset
from unirl.config.require import require
from unirl.config.validation import (
    PrecisionName,
    is_direct_sampling,
    validate_dynamic_dotpaths,
    validate_lora_target_modules,
    validate_offload_contract,
    validate_precision_type,
    validate_rollout_layout,
    validate_training_batch_geometry,
    validate_weight_sync_contract,
)

__all__ = [
    "PrecisionName",
    "build",
    "freeze",
    "is_direct_sampling",
    "materialize",
    "register_config",
    "register_preset",
    "require",
    "validate",
    "validate_dynamic_dotpaths",
    "validate_lora_target_modules",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_rollout_layout",
    "validate_training_batch_geometry",
    "validate_weight_sync_contract",
]


# Walks the whole unirl package and imports every submodule so that
# @register_config decorators (scattered across config/, training/,
# algorithms/, models/, ray/, rollout/, reward/, etc.) populate
# Hydra's ConfigStore. Without this, train.yaml's defaults list
# references unresolved leaves (resume/default, training/plan/default, ...).
#
# Call this explicitly from the train entry, NOT as an import side effect of
# this package: when a Ray worker re-imports unirl through a partially
# loaded module, the walk hits "from X import Y" against an X that is mid-init
# and trips a circular import. Lazy invocation from the @hydra.main caller is
# the only safe time to run it.
def register_all_configs() -> None:
    import importlib
    import logging
    import pkgutil
    import sys

    import unirl

    # Rollout engines (vllm-omni, sglang) and reward scorers carry
    # heavy optional runtime deps (vllm, vllm_omni, sglang, easyocr, …).
    # On a CPU pod or compose-only environment those won't import; their
    # @register_config decorators just don't get exercised in that
    # environment, which is fine — recipes that don't need them still
    # compose. Real (non-optional) ImportError still propagates.
    _OPTIONAL_RUNTIME_DEPS = {
        "vllm",
        "vllm_omni",
        "sglang",
        "easyocr",
        "hpsv3",
        "pickscore",
        "msgspec",
        "ImageReward",
        "mmcv",
        "mmdet",
    }
    _registration_logger = logging.getLogger(__name__)

    for module_info in pkgutil.walk_packages(unirl.__path__, prefix=f"{unirl.__name__}."):
        if module_info.name in sys.modules:
            continue
        try:
            importlib.import_module(module_info.name)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".", 1)[0]
            if missing in _OPTIONAL_RUNTIME_DEPS:
                _registration_logger.debug(
                    "register_all_configs: skipping %s (optional dep %r missing)",
                    module_info.name,
                    missing,
                )
                continue
            raise


__all__ = list(__all__) + ["register_all_configs"]
