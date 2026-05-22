# Weight Sync

`diffusionrl.distributed.weight_sync` owns trainer-side delivery of updated
model weights to rollout engines. It runs only in **dedicated-rollout** modes
(`separate` or `colocate`). Direct sampling (`rollout/engine: trainside`) has
no sync because the same actors do both rollout and train.

## Base Contract

Every backend subclasses `UpdateWeight` (in `base.py`) and implements:

| Method | Called when |
|---|---|
| `connect_rollout_engines()` | Once at startup; opens the NCCL group / IPC sockets / shared filesystem state needed by this backend. |
| `update_weights(*, peft_config=None, base_sync_done=False)` | After every training step that crosses the sync cadence. |

`BucketedUpdateWeight` adds a streaming layer: it walks `raw_state_dict(model)`,
batches parameters into buckets up to `bucket_size` MiB, skips LoRA adapter
tensors, optionally rewrites param names with `param_name_prefix`, and calls
the backend-specific `update_bucket_weights(...)` per bucket. NCCL / IPC /
Tensor inherit this; `checkpoint_path` publishes the full state at once and
does not bucket.

## Registered Backends

All backends register under Hydra group `sync`:

| `sync:` slug | Handler | Engine fit | Transport |
|---|---|---|---|
| `nccl_broadcast` | `UpdateWeightFromDistributed` (`nccl.py`) | SGLang or vLLM-Omni receivers with distributed update RPCs | Temporary NCCL group, `broadcast` per bucket |
| `tensor_payload` | `UpdateWeightFromTensor` (`tensor.py`) | Colocated SGLang or vLLM-Omni receivers with Ray tensor RPCs | Serialized named tensors over Ray RPC; LoRA-only updates can send adapter tensors after base sync |
| `ipc_bucketed` | `UpdateWeightFromIPC` (`ipc.py`) | vLLM-Omni | Bucketed CUDA-IPC over ZMQ; per-stage fan-out for multi-stage engines (HI3 AR + DiT) |
| `checkpoint_path` | `UpdateWeightFromCheckpoint` (`checkpoint.py`) | Any engine that polls a path | Atomic filesystem publish; rollout side loads from disk |

Engine ↔ sync pairing is enforced by
`diffusionrl/config/validation.py::validate_weight_sync_contract`:

- direct sampling → `sync:` must be **absent**;
- dedicated rollout → `sync:` must be **present**;
- `ipc_bucketed` → rollout engine must be vLLM-Omni.

Pick a backend in YAML:

```yaml
defaults:
  - override /rollout/engine: sglang
  - override /sync: nccl_broadcast

sync:
  bucket_size: 256
  target_modules: ["transformer"]
```

## Trigger Path

```text
diffusionrl.train
  -> train_group.setup_weight_sync.remote(sync_cfg=cfg.sync, ...)
       per train actor:
         build(cfg.sync, model=..., rollout_runtime=..., placement_cfg=...)
         handler.connect_rollout_engines()
  -> main loop
     every cfg.run.weight_sync_interval rollouts:
       TrainingWeightSyncMixin.sync_weights_to_rollout()
         handler.update_weights(peft_config=..., base_sync_done=...)
```

`peft_config` is forwarded so LoRA-aware backends can send adapter state with
the metadata expected by the rollout engine. `base_sync_done` lets LoRA-only
paths skip resending already-shipped base weights on subsequent steps.

`weight_version` increments inside `update_weights(...)`. Rollout engines
compare the received version against their last-applied version so a duplicate
sync (e.g. after a resume) is a no-op.

## Transfer Queue — Separate Concern

`diffusionrl.distributed.transfer_queue` is **not** part of weight sync. It
is a data-plane bus for bulky rollout outputs (conditions, latents, rewards)
flowing from rollout actors back to the trainer in separate-deployment mode.
Weight sync is trainer→rollout; transfer queue is rollout→trainer. Don't
conflate them when adding a new sync backend.

## Adding a Sync Backend

1. Subclass `UpdateWeight` (one-shot) or `BucketedUpdateWeight` (streaming).
   Implement `connect_rollout_engines()` and `update_weights(...)` — plus
   `update_bucket_weights(...)` if bucketed.
2. Register the config dataclass under group `sync`:

   ```python
   @register_config(
       group="sync",
       name="my_sync",
       target="my_pkg.MyHandler",
       expand=True,
   )
   @dataclass(frozen=True)
   class MySyncConfig:
       bucket_size: int = 256
       ...
   ```

3. If the backend only works with a specific rollout engine, add a pairing
   check to `validate_weight_sync_contract` so config errors surface before
   Ray actors are created.
4. Make `update_weights(...)` idempotent on `weight_version` to survive
   resume / retry without double-shipping.
