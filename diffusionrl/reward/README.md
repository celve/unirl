# Reward Package

`diffusionrl.reward` owns reward component construction, execution, and
aggregation. Rollout engines generate media; reward components score it and
return per-sample values that rollout control turns into advantages.

## Key Files

| File | Purpose |
|---|---|
| `config.py` | `RewardConfig`, component list, aggregation method, base device |
| `base.py` | scorer/executor interfaces and component spec base class |
| `pipeline.py` | reward execution pipeline |
| `service.py` | reward service construction from configs |
| `reward_service_executor.py` | remote reward-service component |
| `scorers/` | built-in local scorers such as PickScore, HPS, OCR, VideoPickScore |

## Config Shape

Reward config is component-based:

```yaml
reward:
  aggregation_method: weighted_sum
  base_device: cuda
  components:
    - name: pickscore
      weight: 1.0
      batch_size: 8
```

`reward.components` is a polymorphic list. Each component registers a spec in
the `reward/component` ConfigStore group, and the spec target constructs the
runtime scorer or executor.

## Adding a Local Scorer

Define a spec and scorer:

```python
from diffusionrl.config.registration import register_config
from diffusionrl.reward.base import BaseRewardComponentSpec
from diffusionrl.reward.scorers.base_local import BaseLocalRewardScorer
from diffusionrl.types.reward import RewardRequest

class MyRewardScorer(BaseLocalRewardScorer):
    canonical_model_name = "my_reward"

    def __init__(self, *, config: "MyRewardSpec", base_device: str) -> None:
        super().__init__(device=base_device, batch_size=config.batch_size)

    def _load_model(self) -> None:
        ...

    def _compute_model_rewards(self, request: RewardRequest) -> list[float]:
        ...

@register_config(
    group="reward/component",
    name="my_reward",
    target="my_module.MyRewardScorer",
)
class MyRewardSpec(BaseRewardComponentSpec):
    weight: float = 1.0
    batch_size: int = 8
```

Use it from YAML:

```yaml
reward:
  aggregation_method: mean
  base_device: cuda
  components:
    - name: my_reward
      weight: 1.0
      batch_size: 8
```

Prefer `BaseLocalRewardScorer` for in-process model scorers because it provides
device, eager load, `offload()`, and `onload()` behavior. Use a custom executor
or `reward_service_executor.py` when scoring happens out of process.
