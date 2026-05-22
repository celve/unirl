# Training Package

`diffusionrl.training` owns train-side model execution. It receives
`RolloutResp` shards from Ray train actors, replays model stages, computes
losses through slot-keyed algorithms, and performs optimizer steps.

## Key Files

| File | Purpose |
|---|---|
| `stack.py` | `StageTrainStack`, micro-batch slicing, optimizer-step execution |
| `policy.py` | `Policy` protocol, `PolicyBase`, `compose_policy`, chain walking |
| `lora_policy.py` | LoRA injection policy |
| `fsdp_policy.py` | FSDP2 wrapping, grad clipping, checkpoint state dict behavior |
| `ema_policy.py` | EMA shadow policy |
| `nft_lora_policy.py` | NFT dual-adapter policy behavior |
| `execution.py` | offload and gradient execution config |
| `plan.py` | training batch and micro-batch geometry |
| `factories.py` | optimizer and scheduler construction |
| `backends/base.py` | optimizer, scheduler, and topology config schemas |

## Actor Flow

`ray/group/train.py` creates `TrainActor` handles. Each train actor:

1. materializes the selected model bundle;
2. selects the stage named by `cfg.training.policy_source`;
3. composes `cfg.training.policies` around that stage;
4. builds optimizer and scheduler;
5. builds `cfg.algorithms.<slot>` loss objects;
6. trains on `RolloutResp` shards sent by `TrainActorGroup.train(...)`.

## Policy Stack

Policies are trainable-module facades. They wrap a stage or another policy and
expose the surface required by algorithms and checkpoint/sync code.

Example order:

```text
compose_policy(stage, [LoRA, FSDP, EMA])
=> EMA(FSDP(LoRA(stage)))
```

The first config is innermost; the last config is the object returned to the
training stack. Policies mutate or wrap the same underlying module, but expose
different behavior such as trainable-parameter filtering, FSDP state dicts,
EMA stepping, offload/onload, and adapter state extraction.

## Batch Geometry

`training.plan` owns train-side geometry:

- `global_batch_size`: global rollout response size expected by training;
- `local_batch_size`: per-train-actor batch size;
- `local_mini_batch_size`: per-optimizer-step shard before micro slicing;
- `num_updates_per_batch`: optimizer steps per rollout response;
- `micro_batch_size`: memory-oriented slicing inside one optimizer step.

Cross-component validators check divisibility against `training.topology`
before Ray work starts. Keep rollout batch sizing, train topology, and
optimizer-step geometry aligned in experiment YAML.

## Adding a Policy

Add a policy by:

1. implementing a class that follows the `Policy` protocol or extends `PolicyBase`;
2. registering a config with `@register_config(group="training/policy", name=..., target=...)`;
3. accepting the local constructor shape `Policy(config, source)`;
4. adding it to `cfg.training.policies` in an experiment YAML.

Only add a policy when it changes trainable-module behavior. Pure optimizer or
scheduler choices belong under `training.optimizer` or `training.lr_scheduler`.
