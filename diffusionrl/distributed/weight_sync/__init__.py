"""Weight-sync handlers.

The active (v2) handlers are resolved by Hydra via their full ``_target_``
dotted paths and so are not re-exported here:

- full-weight sync: ``diffusionrl.distributed.weight_sync.full.{nccl,ipc,tensor}``
  (``NCCLWeightSync`` / ``IPCWeightSync`` / ``TensorWeightSync``)
- LoRA sync: ``diffusionrl.distributed.weight_sync.lora.LoraWeightSync``

``TrainingWeightSyncMixin.setup_weight_sync`` builds the handler from ``cfg.sync``.
"""

__all__: list[str] = []
