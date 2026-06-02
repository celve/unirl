# Weight Sync

`unirl.distributed.weight_sync` owns trainer-side delivery of updated
model weights to rollout engines. It runs only in **dedicated-rollout** modes
(`separate` or `colocate`). Direct sampling (`rollout/engine: trainside`) has no
sync because the same Remotes do both rollout and train.

## Handlers

Every handler is a `Remote` built on the train slab as a sibling of the FSDP
backend (and, in shared-process colocate, of the rollout engine). It is selected
by its full `_target_` in the `sync:` config block — there is no Hydra group
indirection. Two families:

### LoRA — `lora/`

Two handlers subclass `LoraWeightSyncBase` (`lora/base.py`), which owns the shared
adapter extraction (a train-mesh collective) and the post-load checksum verify;
they differ only in how the adapter reaches the engine.

`LocalLoraWeightSync` (`lora/local.py`) ships the adapter into a co-located engine
via the engine's in-process `set_lora_from_tensors`. `track_prefix` routes a single
track to one child of a `ComposedRolloutEngine`; multi-track training (e.g. PE:
diffusion + ar) registers one handler per track.

`RemoteLoraWeightSync` (`lora/remote.py`) is the same adapter, different transport
topology: when the engines are NOT same-Worker siblings — `DiffusionTrainer`
separate slabs, or HI3's AR / DiT engines on a disjoint GPU partition — the driver
hands rank 0 the engines' `(role, workers)` once (`set_rollout_targets`), then
`sync()` extracts the adapter on the train workers and pushes it from rank 0 to each
engine's `set_lora_from_tensors` over a plain Ray RPC (the adapter and transport stay
inside the handler). The handler does no memory management: separate-slab trainers
call `sync()` (engines on their own GPUs, no contention); a colocate, memory-shared
trainer (HI3) instead splits into `extract()` (run while its engines are asleep —
`state_dict()` gathers the full model to GPU) then `push()` (after offloading the
base and waking the engines), the adapter cached on rank 0 between the two.
`copy=True` routes through the engine's TP>1-safe
byte-copy receiver (`set_lora_from_tensors_copy`), needed for HI3.

### Full-weight — `full/`

`FullWeightSync` (`full/base.py`) provides the transport-agnostic weight walk:
full-tensor materialization (FSDP shard → replicated, a train-mesh collective)
and size-bounded bucketing. `lora_merged=True` folds LoRA deltas into the base
weights before pushing (LoRA-train, serve-merged). Subclasses pick a transport:

| `_target_` module | Class | Transport |
|---|---|---|
| `full/nccl.py` | `NCCLWeightSync` | temporary NCCL group, broadcast per bucket — separate slabs, cross-node capable |
| `full/tensor.py` | `TensorWeightSync` | serialized named-tensor payloads — colocate |
| `full/ipc.py` | `IPCWeightSync` | bucketed CUDA-IPC over ZMQ — colocate, vLLM-Omni |

`payload.py` holds the LoRA handlers' helper: the JSON/Ray-safe PEFT
adapter-config dict.

## Trigger Path

```text
DiffusionTrainer (inside a placement(...) block):
  self.weight_sync = remote_hydra(cfg.sync, backend=self.backend[, rollout=self.rollout])
  self.weight_sync.connect(...)          # one-time transport setup (separate layout)
  ...train loop...
  every weight_sync_interval rollouts:
    self.weight_sync.sync()              # push current weights into the engine(s)
```

The handler reads the trained weights from its `backend` sibling's `.model`. The
LoRA path carries the PEFT config so the engine can reconstruct the
adapter. `weight_version` increments inside the handler so a duplicate sync
(e.g. after a resume) can be made a no-op on the receiver.

## Engine pairing

The rollout-engine receive side lives under
`unirl.rollout.engine.{vllm_omni,sglang}` — e.g. the vLLM-Omni
`ipc_receive_mixin` / `nccl_receive_mixin` worker extensions. Each transport's
sender (`full/*` or `lora`) pairs with the matching receiver on the engine.

## Transfer Queue — Separate Concern

`unirl.distributed.tensor.backend.transfer_queue` is **not** part of weight sync. It is a
data-plane bus for bulky rollout outputs (conditions, latents, rewards) flowing
from rollout actors back to the trainer in separate-deployment mode. Weight sync
is trainer→rollout; transfer queue is rollout→trainer. Don't conflate them when
adding a new sync handler.

## Adding a Sync Handler

1. Subclass `LoraWeightSyncBase` (`lora/base.py`, for a LoRA transport) or
   `FullWeightSync` (`full/base.py`, for a full-weight transport). Implement the
   transport's connection setup and a `sync()` that pushes the current weights.
2. Point a config block's `_target_` at the new class and pass its ctor kwargs
   in the `sync:` block.
3. Make `sync()` idempotent on `weight_version` to survive resume / retry
   without double-shipping.
