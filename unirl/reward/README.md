# Reward Package

`unirl.reward` constructs and runs reward backends. Rollout engines
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
  _target_: unirl.reward.service.RewardService
  backend:
    _target_: unirl.reward.local.pickscore.PickScoreRewardScorer
    base_device: cuda
    config:
      _target_: unirl.reward.local.pickscore.PickScoreSpec
      batch_size: 8
```

For the remote backend, point `backend._target_` at
`unirl.reward.remote.RemoteRewardBackend` with a `RemoteRewardSpec`
(`base_url`, `required_rewards`, …).

## Adding a Local Scorer

```python
from dataclasses import dataclass

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.types.reward import RewardRequest


class MyRewardScorer(LocalRewardBackend):
    canonical_model_name = "my_reward"

    def __init__(self, *, config: "MyRewardSpec", base_device: str) -> None:
        super().__init__(device=base_device, batch_size=config.batch_size)

    def _load_model(self) -> None:
        ...

    def _compute_model_rewards(self, request: RewardRequest) -> list[float]:
        ...


@dataclass
class MyRewardSpec(BaseRewardComponentSpec):
    batch_size: int = 8
```

Wire it in a recipe by `_target_` — `backend._target_` at the scorer, with a
nested `config:` block whose `_target_` is `...MyRewardSpec`.

`LocalRewardBackend` provides device resolution, eager load, `offload()`, and
`onload()`. Use `remote.py` when scoring happens out of process.

## Remote Backend: wire contract & failure semantics

`RemoteRewardBackend` (`remote.py`) is an HTTP client for the standalone
RewardService server (FastAPI + Ray, run on its own GPU node). One backend
multiplexes every `required_rewards` in a single `POST /score` per rollout.

```yaml
reward:
  _target_: unirl.reward.service.RewardService
  backend:
    _target_: unirl.reward.remote.RemoteRewardBackend
    base_device: cpu          # ignored; the backend is HTTP-only
    config:
      _target_: unirl.reward.remote.RemoteRewardSpec
      base_url: ${oc.env:REWARD_SERVICE_URL,http://localhost:8080}
      required_rewards: [hpsv2, clip, ocr]
      reward_weights: {hpsv2: 0.5, clip: 0.3, ocr: 0.2}
      input_kind: image        # "image" (default) or "video"
```

**Wire format.** The single source of truth is RewardService's
`reward_service/schemas.py`. Media rides *inside a history turn* — image and
video share the same shape:

- image: `{"history": [{"text": prompt, "image_b64": ...}], "required_rewards": [...], "metadata": ...}`
- video: `{"history": [{"text": prompt, "video_b64": ...}], ...}`

The server rejects a flat `{"video_b64", "prompt"}` body with HTTP 422. The
single source of truth for the wire schema is the server's Pydantic
`ScoreRequest` in `reward_service.schemas`, in the in-repo `unirl-reward-service/`
subdir (the vendored reward service).

**`input_kind`** selects the modality `score_and_attach` feeds the server. A
video reward (e.g. `videoalign`) is configured as its own component with
`input_kind: video`.

**Failure semantics — loud, never silent.**

- A reward whose value comes back non-finite (`NaN`/`inf`/bool) or `null` (a
  server-side NaN that pydantic serialized to JSON `null`) is flagged as a
  *sample failure* (`success=False`) by `_parse_score_response`, never fed into
  advantage normalization.
- `RewardService.score_and_attach` fail-fasts (raises) on any `success=False`,
  so an infrastructure / inference failure stops the step naming the offending
  reward + sample — it does not silently poison the GRPO group.
- Server side: a scorer signals a *per-item* failure with `NaN` (one bad image
  does not fail the whole batch) and a *whole-reward / config* failure by
  raising.
