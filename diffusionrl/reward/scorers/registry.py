"""Built-in reward scorer registry."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from diffusionrl.reward.base import BaseRewardScorer

_BUILTIN_SCORERS: Dict[str, Tuple[str, str]] = {
    "aesthetic": ("diffusionrl.reward.scorers.aesthetic", "AestheticRewardScorer"),
    "clip": ("diffusionrl.reward.scorers.clip", "ClipRewardScorer"),
    "hpsv2": ("diffusionrl.reward.scorers.hpsv2", "HPSv2RewardScorer"),
    "ocr": ("diffusionrl.reward.scorers.ocr", "OCRRewardScorer"),
    "pickscore": ("diffusionrl.reward.scorers.pickscore", "PickScoreRewardScorer"),
}


def available_builtin_reward_models() -> Tuple[str, ...]:
    """Return built-in LocalRewardScorer model names."""
    return tuple(sorted(_BUILTIN_SCORERS.keys()))


def resolve_builtin_reward_scorer_class(model_name: str) -> Type[BaseRewardScorer]:
    """Resolve a built-in reward scorer class by canonical model name."""
    key = str(model_name or "").strip().lower()
    if key not in _BUILTIN_SCORERS:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: {list(available_builtin_reward_models())}"
        )

    module_name, attr_name = _BUILTIN_SCORERS[key]
    module = importlib.import_module(module_name)
    scorer_cls = getattr(module, attr_name)
    if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
        raise TypeError(
            f"Configured scorer {module_name}.{attr_name} is not a BaseRewardScorer."
        )
    return scorer_cls


__all__ = [
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_class",
]
