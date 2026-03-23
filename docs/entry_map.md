# diffusionRL Entry Map (10 Files)

Goal: let a new contributor find the main training path quickly, without first diving into factory/layout internals.

Read in this order:

1. `diffusionrl/train.py`
   - Sync-only training entry and end-to-end training assembly.
2. `diffusionrl/train_async.py`
   - Async-only training entry; it bootstraps resources itself and keeps `AsyncPipelineRuntime` private to this file.
3. `diffusionrl/config/arguments.py`
   - CLI args, normalization, and runtime validation gates.
4. `diffusionrl/config/build_domain_args.py`
   - Converts flat args into actor/runtime domain configs.
5. `diffusionrl/ray/rollout_manager.py`
   - Rollout-side Ray facade that triggers sampling and delegates business-chain work.
6. `diffusionrl/ray/buffer_actor.py`
   - Ray shell for the rollout/training handoff buffer.
7. `diffusionrl/buffer/core.py`
   - Buffer-side queueing, filtering, reassembly, and dispatch semantics.
8. `diffusionrl/ray/training_group.py`
   - Training actor-group orchestration and high-level train API.
9. `diffusionrl/ray/training_actor.py`
   - Per-actor initialization, backend wiring, and actor-local training lifecycle.
10. `diffusionrl/training/train_executor.py`
   - Training-side execution core (batch partition/update schedule/algorithm-driven loss+backward).

Notes:
- Start with these files before `ray/group_factory.py` and `ray/placement_group.py`.
- `diffusionrl/orchestration/` now holds shared rollout/eval/train workflow helpers, including `request_builder.py`, `rollout_workflow.py`, and `training_workflow.py`.
- Async inflight runtime is private to `diffusionrl/train_async.py`, not exported from `diffusionrl/orchestration/`.
- There is no supported `diffusionrl/runtime/` package; read the semantic packages directly.
- Most algorithm-specific loss/advantage logic now lives under `algorithms/*` after the 10-file pass above.
