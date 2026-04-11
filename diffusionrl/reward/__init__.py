"""Reward subsystem entrypoint."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "AestheticRewardScorer": ("diffusionrl.reward.scorers.aesthetic", "AestheticRewardScorer"),
    "BaseRewardExecutor": ("diffusionrl.reward.base", "BaseRewardExecutor"),
    "BaseLocalRewardScorer": ("diffusionrl.reward.scorers.base_local", "BaseLocalRewardScorer"),
    "BaseRewardScorer": ("diffusionrl.reward.base", "BaseRewardScorer"),
    "ClipRewardScorer": ("diffusionrl.reward.scorers.clip", "ClipRewardScorer"),
    "EditRewardScorer": ("diffusionrl.reward.scorers.editreward", "EditRewardScorer"),
    "HPSv2RewardScorer": ("diffusionrl.reward.scorers.hpsv2", "HPSv2RewardScorer"),
    "HTTPRewardExecutor": ("diffusionrl.reward.http", "HTTPRewardExecutor"),
    "OCRRewardScorer": ("diffusionrl.reward.scorers.ocr", "OCRRewardScorer"),
    "PickScoreRewardScorer": ("diffusionrl.reward.scorers.pickscore", "PickScoreRewardScorer"),
    "compute_rewards_from_rollout_outputs": ("diffusionrl.reward.pipeline", "compute_rewards_from_rollout_outputs"),
    "RewardComponentSpec": ("diffusionrl.reward.schema", "RewardComponentSpec"),
    "RewardDefinition": ("diffusionrl.reward.schema", "RewardDefinition"),
    "RewardExecutionPlan": ("diffusionrl.reward.schema", "RewardExecutionPlan"),
    "RewardProviderConfig": ("diffusionrl.reward.schema", "RewardProviderConfig"),
    "read_rewards": ("diffusionrl.reward.pipeline", "read_rewards"),
    "score_and_attach_rewards": ("diffusionrl.reward.pipeline", "score_and_attach_rewards"),
    "RewardSchema": ("diffusionrl.reward.schema", "RewardSchema"),
    "RewardService": ("diffusionrl.reward.service", "RewardService"),
    "VideoRewardScorer": ("diffusionrl.reward.scorers.video", "VideoRewardScorer"),
    "resolve_reward_input_kind": ("diffusionrl.reward.pipeline", "resolve_reward_input_kind"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


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
