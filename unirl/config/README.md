# Config Package

`unirl.config` owns the typed Hydra surface. Config dataclasses are
registered near the code that consumes them, then imported by
`register_all_configs()` before Hydra composes the chosen `conf/<recipe>.yaml`.

## Main APIs

| API | Purpose |
|---|---|
| `register_config` | Register a dataclass schema or component target in Hydra ConfigStore |
| `register_preset` | Register a preconfigured dataclass instance |
| `validate` | Materialize structured config and fail fast on schema errors |
| `build` | Instantiate a registered `_target_` component |
| `freeze` | Seal composed config nodes except those explicitly marked mutable |
| `require` | Small helper for readable validation errors |

## Registration Pattern

Use `@register_config` for typed component configs:

```python
from unirl.config.registration import register_config

@register_config(
    group="rollout/engine",
    name="my_engine",
    target="my_pkg.MyRolloutEngine",
)
class MyEngineConfig:
    batch_size: int = 8
```

The `target` class should follow the local constructor convention:

```python
class MyRolloutEngine:
    def __init__(self, *, config: MyEngineConfig, **deps) -> None:
        ...
```

Set `expand=True` only for third-party classes that expect flat keyword
arguments instead of `config=<ConfigCls>`.

## Validation Layers

Validation is split into two layers:

- per-dataclass `__post_init__` checks for local field invariants;
- cross-component validators in `validation.py` for contracts spanning
  rollout engine, sync, training geometry, LoRA, offload, and dotpaths.

`unirl.train` runs cross-component validators before any Ray actor is
created. This keeps config mistakes cheap to catch.

## Mutable Config

Most composed config is frozen after validation. Mark a schema
`mutable=True` only when runtime materialization must write back into that
node. The main example is model LoRA target materialization, where the chosen
model bundle may fill `cfg.model.lora_target_modules` if YAML omitted it.
