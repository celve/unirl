# Reward Package

`diffusionrl.reward` constructs and runs reward backends. Rollout engines
generate media; a reward backend scores it and returns per-sample values that
the trainer turns into advantages.

## Structure

A reward is exactly one **backend** — either a local in-process scorer or the
remote RewardService HTTP client — held by `RewardService`.

| File | Purpose |
|---|---|
| `base.py` | `RewardBackend` ABC + `BaseRewardComponentSpec` |
| `service.py` | `RewardService`: holds one backend; scores a `RolloutTrack` via `score_and_attach` |
| `remote.py` | `RemoteRewardBackend`: HTTP client for the remote RewardService server |
| `local/` | local in-process backends (`LocalRewardBackend` + PickScore, HPS, OCR, VideoPickScore, …) |

## Config

A reward is wired via Hydra `_target_`:

```yaml
reward:
  _target_: diffusionrl.reward.service.RewardService
  backend:
    _target_: diffusionrl.reward.local.pickscore.PickScoreRewardScorer
    base_device: cuda
    config:
      _target_: diffusionrl.reward.local.pickscore.PickScoreSpec
      batch_size: 8
```

For the remote backend, point `backend._target_` at
`diffusionrl.reward.remote.RemoteRewardBackend` with a `RemoteRewardSpec`
(`base_url`, `required_rewards`, …).

## Adding a Local Scorer

```python
from diffusionrl.config.registration import register_config
from diffusionrl.reward.base import BaseRewardComponentSpec
from diffusionrl.reward.local.base import LocalRewardBackend
from diffusionrl.types.reward import RewardRequest


class MyRewardScorer(LocalRewardBackend):
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
    target="diffusionrl.reward.local.my_reward.MyRewardScorer",
)
class MyRewardSpec(BaseRewardComponentSpec):
    batch_size: int = 8
```

`LocalRewardBackend` provides device resolution, eager load, `offload()`, and
`onload()`. Use `remote.py` when scoring happens out of process.
