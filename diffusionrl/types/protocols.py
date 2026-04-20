from typing import List, Protocol

import ray

from diffusionrl.transfer.buffer import BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse


class GenerateGroup(Protocol):
    def generate(self, request: RolloutRequest) -> RolloutResponse: ...

    def generate_async(self, request: RolloutRequest) -> ray.ObjectRef: ...

    def generate_buffered(self, request: RolloutRequest) -> List[BufferHandle]: ...
