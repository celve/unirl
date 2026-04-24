from dataclasses import dataclass

from diffusionrl.types.prompts import Prompts
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.utils.batched import Batched, concat_field, shared_field


@dataclass
class RolloutRequest(Batched):
    """Lightweight request contract shared across rollout stages."""

    prompts: Prompts = concat_field()
    sampling_params: SamplingParams = shared_field()
