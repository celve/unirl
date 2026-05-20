"""New-protocol SGLang rollout engine.

This package houses the rewrite of ``diffusionrl/samplers/sglang/`` onto the
new :class:`diffusionrl.rollout.engine.base.BaseRolloutEngine` protocol. It
coexists with the legacy package — recipes opt into it via
``cfg.rollout.engine=sglang_new``.
"""

from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.engine import SGLangRolloutEngine

__all__ = ["SGLangRolloutEngine", "SGLangEngineConfig"]
