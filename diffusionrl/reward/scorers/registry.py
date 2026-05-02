"""Built-in reward scorer / spec registry.

Two parallel maps keyed by ``model_name``:

- ``_BUILTIN_SCORERS`` — model_name → scorer class. Used by composite scorers
  (e.g. ``VideoRewardScorer``) that need to instantiate an inner frame scorer.
- ``_BUILTIN_SPECS`` — model_name → spec dataclass. Same composite case needs
  to materialize a default inner Spec to feed the scorer's
  ``__init__(*, config, base_device)`` entry point.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from diffusionrl.reward.base import BaseRewardComponentSpec, BaseRewardScorer

_BUILTIN_SCORERS: Dict[str, Tuple[str, str]] = {
    "aesthetic": ("diffusionrl.reward.scorers.aesthetic", "AestheticRewardScorer"),
    "clip": ("diffusionrl.reward.scorers.clip", "ClipRewardScorer"),
    "hpsv2": ("diffusionrl.reward.scorers.hpsv2", "HPSv2RewardScorer"),
    "ocr": ("diffusionrl.reward.scorers.ocr", "OCRRewardScorer"),
    "pickscore": ("diffusionrl.reward.scorers.pickscore", "PickScoreRewardScorer"),
    "hpsv3": ("diffusionrl.reward.scorers.hpsv3", "HPSv3RewardScorer"),
    "image_reward": ("diffusionrl.reward.scorers.image_reward", "ImageRewardScorer"),
}

_BUILTIN_SPECS: Dict[str, Tuple[str, str]] = {
    "aesthetic": ("diffusionrl.reward.scorers.aesthetic", "AestheticSpec"),
    "clip": ("diffusionrl.reward.scorers.clip", "ClipSpec"),
    "hpsv2": ("diffusionrl.reward.scorers.hpsv2", "HPSv2Spec"),
    "ocr": ("diffusionrl.reward.scorers.ocr", "OCRSpec"),
    "pickscore": ("diffusionrl.reward.scorers.pickscore", "PickScoreSpec"),
    "hpsv3": ("diffusionrl.reward.scorers.hpsv3", "HPSv3Spec"),
    "image_reward": ("diffusionrl.reward.scorers.image_reward", "ImageRewardSpec"),
}


def available_builtin_reward_models() -> Tuple[str, ...]:
    """Return built-in reward model names."""
    return tuple(sorted(_BUILTIN_SCORERS.keys()))


def _resolve_builtin_reward_entry(model_name: str) -> Tuple[str, str]:
    key = str(model_name or "").strip().lower()
    if key not in _BUILTIN_SCORERS:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(available_builtin_reward_models())}")
    return _BUILTIN_SCORERS[key]


def _resolve_builtin_spec_entry(model_name: str) -> Tuple[str, str]:
    key = str(model_name or "").strip().lower()
    if key not in _BUILTIN_SPECS:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(available_builtin_reward_models())}")
    return _BUILTIN_SPECS[key]


def resolve_builtin_reward_scorer_path(model_name: str) -> str:
    """Resolve the Python dotpath of a built-in reward scorer."""
    module_name, attr_name = _resolve_builtin_reward_entry(model_name)
    return f"{module_name}.{attr_name}"


def resolve_builtin_reward_scorer_class(model_name: str) -> Type[BaseRewardScorer]:
    """Resolve a built-in reward scorer class by canonical model name."""
    module_name, attr_name = _resolve_builtin_reward_entry(model_name)
    module = importlib.import_module(module_name)
    scorer_cls = getattr(module, attr_name)
    if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
        raise TypeError(f"Configured scorer {module_name}.{attr_name} is not a BaseRewardScorer.")
    return scorer_cls


def resolve_builtin_reward_spec_class(model_name: str) -> Type[BaseRewardComponentSpec]:
    """Resolve a built-in reward Spec class by canonical model name."""
    module_name, attr_name = _resolve_builtin_spec_entry(model_name)
    module = importlib.import_module(module_name)
    spec_cls = getattr(module, attr_name)
    if not isinstance(spec_cls, type) or not issubclass(spec_cls, BaseRewardComponentSpec):
        raise TypeError(f"Configured spec {module_name}.{attr_name} is not a BaseRewardComponentSpec.")
    return spec_cls


__all__ = [
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_path",
    "resolve_builtin_reward_scorer_class",
    "resolve_builtin_reward_spec_class",
]
