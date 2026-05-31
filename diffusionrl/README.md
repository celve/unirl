# Code Architecture

This package is organized around one runtime loop:

```text
diffusionrl.train
  -> register and validate Hydra config
  -> create Ray placement
  -> create RolloutActorGroup and TrainActorGroup
  -> RolloutPipeline runs rollout, reward, advantage, train, sync
```

At a topological level:

```text
           driver (diffusionrl.train + RolloutPipeline)
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
   RolloutActorGroup                    TrainActorGroup
   (engine: trainside |                 (model bundle + policy stack +
    sglang | vllm_omni)                  stage algorithms + optimizer)
            │                                   ▲
            │   RolloutReq / RolloutResp        │
            └──────────► reward ───► advantage ─┘
                              │
                              ▼ (dedicated rollout modes only)
                       diffusionrl.distributed.weight_sync
                       (nccl_broadcast | tensor_payload |
                        ipc_bucketed | checkpoint_path)
```

The code intentionally separates driver control flow, actor orchestration,
rollout engines, training policy composition, and algorithm loss math.

## Module Map

| Path | Responsibility |
|---|---|
| `train.py` | Hydra entrypoint and outer training lifecycle |
| `config/` | ConfigStore registration, instantiation, readonly sealing, cross-component validation |
| `ray/` | Ray actor implementations, actor groups, placement, and actor utility code |
| `rollout/` | Driver-side rollout pipeline, request planning, rollout engine contracts |
| `training/` | Train actor stack, composable policy chain, optimizer-step execution |
| `algorithms/` | Stage-driven loss computation and driver-side rollout control |
| `models/` | Per-model bundles, stages, conditions, text/vision/vae helpers |
| `reward/` | Reward service, in-process scorers, component aggregation |
| `distributed/` | Weight sync and transfer queue implementations |
| `types/` | Shared typed data contracts: requests, responses, conditions, segments, rewards |
| `data/` | Data source and dataset readers |
| `sde/` | SDE strategy rules and runtime kernels |
| `utils/` | Logging, dtype, media, timing, checkpoint, and misc helpers |

## Runtime Data Flow

1. `diffusionrl.train` composes config and runs validators.
2. `placement` describes which Ray bundle each train/rollout actor should use.
3. `RolloutActorGroup` dispatches typed `RolloutReq` objects to rollout actors.
4. Rollout engines produce `RolloutResp`, whose `tracks[slot]` carry conditions, segments, rewards, and media previews.
5. `RolloutPipeline` aggregates responses and computes rollout-control outputs such as advantages.
6. `TrainActorGroup.train(...)` shards `RolloutResp` across train actors.
7. Each train actor owns a model bundle, a policy stack, stage algorithms, optimizer, and scheduler.
8. Dedicated-rollout modes sync trainer weights back to rollout actors.

## Important Boundaries

- Driver-side rollout control is `cfg.algorithm` and lives in `algorithms/rollout_control.py`.
- Train-side loss objects are `cfg.algorithms.<slot>` and live in `algorithms/`.
- `RolloutReq` and `RolloutResp` are the rollout/training boundary.
- `Policy` objects are trainable-module facades; they wrap model stages with LoRA, FSDP, EMA, or NFT behavior.
- A rollout engine owns sampling backend details; the rest of the system should talk through typed request/response objects.
- Config classes live near the implementation that consumes them.

## Deeper Module Docs

- `config/README.md`: Hydra groups, config ownership, validators.
- `ray/README.md`: actor groups, placement, and orchestration boundaries.
- `rollout/README.md`: rollout modes, engines, request/response flow.
- `training/README.md`: train actors, policy stack, optimizer-step geometry.
- `algorithms/README.md`: rollout control vs train-side algorithms.
- `reward/README.md`: reward components and custom scorers.
- `models/README.md`: model bundle and per-model package contracts.
