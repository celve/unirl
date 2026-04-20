from dataclasses import dataclass
from typing import Any, Dict, List
import copy

from diffusionrl.types.prompts import Prompts
from diffusionrl.utils.batched import Batched, shared_field, concat_field
from diffusionrl.types.sampling import SamplingParams


@dataclass
class RolloutRequest(Batched):
    """Lightweight request contract shared across rollout stages."""

    prompts: Prompts = concat_field()
    sampling_params: SamplingParams = shared_field()