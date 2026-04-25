from dataclasses import dataclass

from diffusionrl.types.prompts import Prompts
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.utils.batched import Batched, concat_field, shared_field


@dataclass
class RolloutRequest(Batched):
    """Lightweight request contract shared across rollout stages.

    ``collect_media_preview`` / ``media_max_items`` are driver-supplied
    control-plane knobs: when the former is true, the actor-side rollout
    pipeline captures up to ``media_max_items`` decoded PIL images (plus
    their prompts and rewards) into ``RolloutSamples.media_preview`` after
    reward scoring, and then drops the full decoded-image lists so only the
    small preview payload traverses Ray back to the driver.
    """

    prompts: Prompts = concat_field()
    sampling_params: SamplingParams = shared_field()
    collect_media_preview: bool = shared_field(default=False)
    media_max_items: int = shared_field(default=8)
