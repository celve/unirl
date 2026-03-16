# diffusionRL Entry Map (8 Files)

Goal: let a new contributor find the main training path quickly, without first diving into factory/layout internals.

Read in this order:

1. `diffusionrl/train.py`
   - Main entry and end-to-end training loop (rollout -> train -> sync).
2. `diffusionrl/config/arguments.py`
   - CLI args, normalization, and runtime validation gates.
3. `diffusionrl/config/build_domain_args.py`
   - Converts flat args into actor/runtime domain configs.
4. `diffusionrl/ray/rollout_manager.py`
   - Sampling, reward/advantage computation, training-batch assembly.
5. `diffusionrl/ray/buffer_actor.py`
   - Rollout/training handoff buffer and alignment/queue semantics.
6. `diffusionrl/ray/training_group.py`
   - Training actor-group orchestration and high-level train API.
7. `diffusionrl/ray/training_actor.py`
   - Per-actor initialization, backend wiring, and step execution.
8. `diffusionrl/runtime/training/backward_train_step.py`
   - Backward/update core path (loss forward/backward/optimizer step).

Notes:
- Start with these files before `ray/group_factory.py` and `ray/placement_group.py`.
- Most algorithm-specific logic should be tracked via `algorithms/*` and `losses/*` after the 8-file pass above.
