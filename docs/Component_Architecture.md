# DiffusionRL Component Architecture

This document reflects the current three-block structure as of 2026-03-22:

- user-facing extension surface
- readable workflow layer
- internal kernel subsystems

See also:

- `docs/Package_Contracts.md` for package-level dependency and ownership rules.

## 1. Structure

### User Surface

- `diffusionrl_plugins/`
  - official external extension entry
- `diffusionrl/algorithms/`
  - builtin algorithms and reference implementations
- `diffusionrl/reward/`
  - builtin reward executors and reward-side helpers
- `diffusionrl/config/`
  - config entry, normalization, and contract resolution

### Workflow Layer

- `diffusionrl/orchestration/rollout_workflow.py`
  - owns `sample -> reward -> advantage -> assemble`
- `diffusionrl/orchestration/request_builder.py`
  - expands prompt batches into rollout requests and request splits
- `diffusionrl/orchestration/eval_workflow.py`
  - owns eval-side sample/reward flow
- `diffusionrl/orchestration/training_workflow.py`
  - owns train-side business chain after batch materialization

Async inflight rollout state is currently private to `diffusionrl/train_async.py`.

### Kernel Subsystems

- `diffusionrl/ray/`
  - actor entrypoints, groups, placement, Ray-specific control plumbing
- `diffusionrl/buffer/`
  - rollout buffer queueing, filtering, group reassembly, store abstraction
- `diffusionrl/training/`
  - train executor, update schedule, backend contracts and implementations
- `diffusionrl/distributed/`
  - distributed coordination semantics and sync protocols
  - may bind the active transport/runtime boundary when coordination requires it
  - does not own Ray actor/group/placement definitions

## 2. Driver Boundary

`diffusionrl/train.py` and `diffusionrl/train_async.py` are parallel training entrypoints.

It is allowed to know:

- placement groups
- actor groups
- rollout buffer actor creation
- weight sync coordinator wiring
- outer loop phase boundaries

It should not own:

- rollout reward logic
- batch assembly logic
- optimizer-update slicing logic
- actor-local lifecycle internals

## 3. Main Flow

### Synchronous

1. `train.py` resolves config and creates Ray/kernel components.
2. `RolloutManager` prepares prompt batches and calls `RolloutWorkflow`.
3. `RolloutWorkflow` executes `sample -> reward -> advantage -> assemble`.
4. `BufferActor` delegates queueing/reassembly to `buffer.BufferRuntime`.
5. `TrainingActor` materializes a batch and delegates business execution to `TrainingWorkflow`.
6. `TrainExecutor` runs the explicit update plan.
7. `WeightSyncCoordinator` performs rollout/training synchronization when required.

### Asynchronous

1. `train_async.py` resolves config and creates its own Ray/kernel components.
2. `train_async.py` keeps a small private `AsyncPipelineRuntime` to track inflight rollout futures.
3. rollout production overlaps with training.
4. weight sync is still forced at explicit generation boundaries.

## 4. Current Design Rules

- Driver can know Ray.
- Workflow code should not depend on Ray actor handles or placement details.
- Ray files do not own rollout business chains.
- Buffer semantics live in `diffusionrl/buffer/`, not in Ray actor files.
- Small async driver state may live in `train_async.py` when it is private to the async loop.
- `diffusionrl/distributed/` may use the active transport/runtime boundary for
  coordination, but it does not own actor creation, placement, or workflow code.
- No `diffusionrl/runtime/` package remains; historical runtime code has been moved into semantic packages.
