# Config Package

`unirl.config` holds small helpers for the flat-recipe config flow. Config
classes are plain `@dataclass`es defined next to the code that consumes them;
recipes wire them by their `_target_` dotpath (no ConfigStore).

## Main APIs

| API | Purpose |
|---|---|
| `require` | One-line precondition helper for `__post_init__` |
| `validate_*` / `PrecisionName` | Cross-component validators + precision helpers (`validation.py`) |

## Config Pattern

Define a config as a plain dataclass and point the recipe's `_target_` at the
component; the component takes `config=<ConfigCls>`:

```python
from dataclasses import dataclass

@dataclass
class MyEngineConfig:
    batch_size: int = 8

class MyRolloutEngine:
    def __init__(self, *, config: MyEngineConfig, **deps) -> None:
        ...
```

Recipe:

```yaml
rollout:
  _target_: my_pkg.MyRolloutEngine
  config:
    _target_: my_pkg.MyEngineConfig
    batch_size: 8
```

The worker walker (`distributed/group/worker.py::_resolve_init_kwargs`)
constructs nested `_target_` blocks by `get_method(_target_)(**fields)`;
`hydra.utils.instantiate` covers the driver-side cases.

## Validation Layers

- per-dataclass `__post_init__` checks for local field invariants;
- cross-component validators in `validation.py` for contracts spanning rollout
  engine, sync, training geometry, LoRA, offload, and dotpaths.
