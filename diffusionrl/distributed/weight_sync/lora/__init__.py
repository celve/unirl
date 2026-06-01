"""LoRA weight-sync handlers for the v2 trainer.

- ``LocalLoraWeightSync``  — colocate, same-Worker sibling, in-process push.
- ``RemoteLoraWeightSync`` — cross-process Ray push (separate slabs + HI3).

Both subclass ``LoraWeightSyncBase`` and are referenced from configs via ``_target_``
(e.g. ``diffusionrl.distributed.weight_sync.lora.RemoteLoraWeightSync``).
"""

from diffusionrl.distributed.weight_sync.lora.base import LoraWeightSyncBase
from diffusionrl.distributed.weight_sync.lora.local import LocalLoraWeightSync
from diffusionrl.distributed.weight_sync.lora.remote import RemoteLoraWeightSync

__all__ = ["LoraWeightSyncBase", "LocalLoraWeightSync", "RemoteLoraWeightSync"]
