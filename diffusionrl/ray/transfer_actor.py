from collections import deque
from collections.abc import Buffer
from typing import Deque, Dict, Optional

from diffusionrl.types.response import RolloutResponse, RolloutResponseMeta
import ray

from diffusionrl.transfer.buffer import BufferHandle


@ray.remote(num_cpus=1, num_gpus=0)
class TransferActor:
    def __init__(self, group_size: int):
        self.pendings: Dict[RolloutResponseMeta, BufferHandle[RolloutResponse]] = Buffer()
        self.readies: Deque[BufferHandle[RolloutResponseMeta, RolloutResponse]] = deque()
        self.group_size = group_size
    
    def _create(self, handle: BufferHandle): 
        if handle.key.batch_size > self.group_size:
            raise ValueError(f"Group size exceeded for key {handle.key}")
        elif handle.key.batch_size == self.group_size:
            self.readies.append(handle)
        else:
            self.pendings[handle.key] = handle
    
    def push(self, external_handle: BufferHandle) -> None:
        if external_handle.key in self.pendings:
            old_handle = self.pendings.pop(external_handle.key)
            new_handle = external_handle.transfer_to(old_handle.actor_handle)
            self._create(new_handle)
        else: 
            self._create(external_handle)
    
    def pop(self) -> Optional[BufferHandle]:
        if len(self.readies) == 0:
            return None
        return self.readies.popleft()