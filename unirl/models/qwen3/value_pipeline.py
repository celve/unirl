"""Training-side stage holder for the independent Qwen3 value critic."""

from __future__ import annotations

from typing import Optional

from unirl.models.types.pipeline import Pipeline
from unirl.types.sample import Sample

from .config import Qwen3ValueConfig
from .value import Qwen3ValueStage
from .value_bundle import Qwen3ValueBundle


class Qwen3ValuePipeline(Pipeline):
    """Expose the critic as ``pipeline.value`` for train-stack wiring.

    This object is intentionally not a rollout pipeline.  It only follows the
    existing bundle/pipeline/stage construction shape so generic training code
    can resolve ``stage_attr='value'`` and share the already-loaded bundle.
    """

    def __init__(
        self,
        *,
        bundle: Qwen3ValueBundle,
        value: Optional[Qwen3ValueStage] = None,
        autocast_precision: str = "bf16",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.value = value or Qwen3ValueStage(
            model=bundle,
            autocast_precision=autocast_precision,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Qwen3ValueBundle,
        *,
        autocast_precision: str = "bf16",
    ) -> "Qwen3ValuePipeline":
        return cls(bundle=bundle, autocast_precision=autocast_precision)

    @classmethod
    def from_config(cls, config: Qwen3ValueConfig) -> "Qwen3ValuePipeline":
        bundle = Qwen3ValueBundle.from_config(config)
        return cls.from_bundle(bundle, autocast_precision=config.autocast_precision)

    def generate(self, sample: Sample) -> Sample:
        raise RuntimeError(
            "Qwen3ValuePipeline is training-only and cannot generate rollouts; use the actor Qwen3Pipeline for rollout"
        )


__all__ = ["Qwen3ValuePipeline"]
