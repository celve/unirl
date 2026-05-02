import uuid
from dataclasses import dataclass
from typing import Dict

from ray.actor import ActorHandle

from diffusionrl.utils.batched import Batched


@dataclass
class BufferHandle:
    id: str
    key: Batched
    actor_handle: ActorHandle

    def transfer_to(self, actor_handle: ActorHandle) -> None:
        actor_handle.put_buffer.remote(self.id, self.actor_handle.get_buffer.remote(self.id))
        self.actor_handle.release_buffer.remote(self.id)
        self.actor_handle = actor_handle


class Buffer:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mappings: Dict[str, Batched] = {}
        self.actor_handle: ActorHandle = None

    def put_buffer(self, key: Batched, value: Batched) -> BufferHandle:
        id = str(uuid.uuid4())
        self.mappings[id] = value
        return BufferHandle(id=id, key=key, actor_handle=self.actor_handle)

    def concat_buffer(self, handle1: BufferHandle, handle2: BufferHandle) -> BufferHandle:
        id = str(uuid.uuid4())
        value1 = self.pop_buffer(handle1)
        value2 = self.pop_buffer(handle2)
        key = type(handle1.key).concat([handle1.key, handle2.key])
        value = type(value1).concat([value1, value2])
        self.mappings[id] = value
        return BufferHandle(id=id, key=key, actor_handle=self.actor_handle)

    def get_buffer(self, handle: BufferHandle) -> Batched:
        return self.mappings[handle.id]

    def pop_buffer(self, handle: BufferHandle) -> Batched:
        return self.mappings.pop(handle.id)
