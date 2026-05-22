# Ray Package

`diffusionrl.ray` owns distributed orchestration. It should not contain model
loss math, reward math, or backend-specific sampling logic; it creates actors,
routes typed data, and manages control-plane operations.

## Key Files

| File | Purpose |
|---|---|
| `placement.py` | Ray placement group layout and train/rollout actor placement metadata |
| `rollout_actor.py` | single rollout worker implementation |
| `train_actor.py` | single train worker implementation |
| `group/base.py` | shared actor-group scatter/gather helpers |
| `group/rollout.py` | rollout actor group creation and dispatch |
| `group/train.py` | train actor group creation, sharding, train dispatch, weight-sync control |
| `mixins/` | reusable actor behavior for rollout pipeline and weight sync |
| `utils/` | stateless helpers for GPU, node, and network discovery |

## Layering

Use this boundary when editing Ray code:

- actor files own one worker's lifecycle and RPC methods;
- group files own multi-actor orchestration and balanced dispatch;
- `placement.py` owns resource layout and colocate bundle decisions;
- `rollout/` owns request planning and rollout engine behavior;
- `training/` owns model replay, policy composition, and optimizer execution.

Direct sampling is represented by `RolloutActorGroup.from_train_group(...)`:
the rollout group adopts train actor handles instead of spawning separate
rollout actors. Dedicated rollout modes create real rollout actors and require
a weight-sync variant.

## Common Failure Modes

- Train actor count must match `training.topology.actor_count` when that field
  is set.
- Every train actor must receive at least one sample; otherwise FSDP
  collectives can hang.
- Colocate mode uses fractional Ray GPU claims to avoid scheduling deadlocks
  when rollout and train actors share bundles.
- Rank-0 train actor rebroadcasts its actual master address and port after
  startup so multinode collectives do not rely on a driver-local address.
