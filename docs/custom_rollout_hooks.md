# Custom Rollout Hooks

diffusionRL provides three pluggable hooks that control the rollout pipeline:

| Hook | Config key | Default | Signature |
|------|-----------|---------|-----------|
| **rollout function** | `rollout_function_dotpath` | `diffusionrl.rollout.default_rollout.generate_rollout` | `(*, services, reward_hook, context) -> RolloutFunctionResult` |
| **eval function** | `eval_function_dotpath` | `diffusionrl.rollout.default_rollout.evaluate_rollout` | `(*, services, reward_hook, rollout_id) -> Dict[str, Any]` |
| **reward hook** | `reward_hook_dotpath` | `diffusionrl.rollout.default_rollout.score_rewards_hook` | `(*, services, request, samples, rollout_id) -> RewardHookResult` |

These are set via YAML config or CLI:

```yaml
rollout_function_dotpath: "my_project.rollout.generate_rollout"
eval_function_dotpath: "my_project.rollout.evaluate_rollout"
reward_hook_dotpath: "my_project.rewards.my_reward_hook"
```

```bash
python -m diffusionrl.train \
    --config scripts/my_config.yaml \
    --reward-hook-dotpath my_project.rewards.gemini_reward
```

By default, the active sampling actor precomputes scalar rewards and drops decoded
media unless the rollout explicitly asks to keep previews. Driver-side reward hooks
should therefore prefer `read_rewards(...)` when they only need scalar rewards.
If a hook needs raw images or videos, pair it with a custom rollout function that
forces `collect_media_preview=True`, or move the scorer into actor-side
`--reward.reward-dotpath`.

## Architecture overview

The driver (train.py) calls these hooks in the following order:

```
                         rollout_function(services, reward_hook, context)
                           │
                           │  internally calls:
                           │    services.load_prompt_batch()
                           │    services.plan_request_batches()
                           │    services.execute_sampling_request()  → actor_group.generate()
                           │    reward_hook(services, request, samples, rollout_id)
                           │
                           ▼
                        RolloutFunctionResult
                          .request           ← RolloutRequest
                          .sampler_outputs   ← List[RolloutSamples]
                          .rewards           ← Tensor
                         .component_rewards ← Dict
                           │
  ┌────────────────────────┤  (driver does these explicitly)
  │                        │
  ▼                        ▼
services.compute_advantages()    algorithm.assemble_training_batch()
                                          │
                                          ▼
                                    TrainingBatch → buffer → training actors
```

Key design: the rollout function returns **intermediate products** (request, samples, rewards),
not a finished TrainingBatch. Advantage computation and batch assembly happen on the driver
side, outside the rollout function. This means custom rollout functions never need to know
about training batch types.

## When to customize which hook

| I want to... | Customize |
|---|---|
| Change how rewards are computed (e.g., reshape precomputed rewards, or score preserved media on the driver) | `reward_hook_dotpath` |
| Change how prompts are loaded or how requests are built (e.g., curriculum learning, force media retention) | `rollout_function_dotpath` |
| Change evaluation logic (e.g., FID computation, different eval dataset) | `eval_function_dotpath` |

## Example 1: Custom reward hook (scalar post-processing)

The simplest extension point. Read the precomputed actor-side rewards and reshape
them — no decoded images needed:

```python
# my_project/rewards.py
import torch
from diffusionrl.reward.pipeline import read_rewards
from diffusionrl.rollout.base_types import RewardHookResult

def clipped_reward_hook(*, services, request, samples, rollout_id):
    """Read precomputed rewards, apply clipping and a length bonus."""
    rewards, component_rewards = read_rewards(sampler_outputs=samples)
    # Clip outliers and add a small bonus for longer prompts
    rewards = rewards.clamp(-1.0, 1.0)
    for i, prompt in enumerate(request.prompts):
        if len(prompt.split()) > 20:
            rewards[i] += 0.1
    return RewardHookResult(
        rewards=rewards,
        component_rewards=component_rewards,
    )
```

```yaml
reward_hook_dotpath: "my_project.rewards.clipped_reward_hook"
```

## Example 2: Custom rollout function with driver-side scoring

If your reward hook needs decoded images (e.g., calling an external API), you must
pair it with a custom rollout function that forces `collect_media_preview=True` so
the actor keeps media instead of dropping it after actor-side scoring.

The default pipeline is built from three composable building blocks:

- `prepare_default_rollout_plan(services)` - load prompts, plan request batches
- `execute_request_batches(...)` - dispatch typed requests and aggregate outputs
- `finalize_default_rollout(services, reward_hook, context, ...)` - score rewards, build result

You can replace any piece while reusing the rest:

```python
# my_project/rollout.py
import torch
from diffusionrl.rollout.base_types import RewardHookResult
from diffusionrl.rollout.default_rollout import finalize_default_rollout
from diffusionrl.rollout.primitives import (
    build_sampler_output_validator,
    execute_request_batches,
)

def media_preserving_rollout(*, services, reward_hook, context):
    """Custom rollout that keeps decoded media for driver-side reward hooks."""
    batch = services.load_prompt_batch()
    request_batches = services.plan_request_batches(
        batch=batch,
        samples_per_prompt=services.samples_per_prompt,
    )
    request, sampler_outputs = execute_request_batches(
        request_batches=request_batches,
        rollout_id=context.rollout_id,
        execute_sampling_request=lambda request: services.execute_sampling_request(
            request=request,
            sde_indices=context.sde_indices,
            requirements=services.sampling_requirements,
            sampling_overrides={
                "collect_media_preview": True,  # keep decoded images
            },
        ),
        validate_sampler_outputs=build_sampler_output_validator(
            requirements=services.sampling_requirements,
            validation_config=services.sampler_validation_config,
        ),
    )
    return finalize_default_rollout(
        services=services,
        reward_hook=reward_hook,
        context=context,
        batch=batch,
        request_batches=request_batches,
        request=request,
        sampler_outputs=sampler_outputs,
    )


def gemini_reward_hook(*, services, request, samples, rollout_id):
    """Score generated images using Gemini API (requires media_preserving_rollout)."""
    rewards = []
    for output in samples:
        decoded_images = output.aux.get("decoded_images", [])
        for image in decoded_images:
            score = call_gemini_api(image, request.prompts[len(rewards)])
            rewards.append(score)
    return RewardHookResult(
        rewards=torch.tensor(rewards, dtype=torch.float32),
    )
```

```yaml
rollout_function_dotpath: "my_project.rollout.media_preserving_rollout"
reward_hook_dotpath: "my_project.rollout.gemini_reward_hook"
```

## Example 3: Custom eval function

```python
# my_project/eval.py
from diffusionrl.rollout.service_interface import build_eval_request_batch

def fid_evaluation(*, services, reward_hook, rollout_id):
    """Evaluate using FID score instead of reward."""
    request_batch = build_eval_request_batch(
        data_source=services.data_source,
        prompt_batch_size=services.prompt_batch_size,
        evaluation_settings=services.evaluation_settings,
    )
    request = services.build_request(
        batch=request_batch,
        samples_per_prompt=1,
    )
    sampler_outputs = services.execute_sampling_request(
        request=request,
        sde_indices=None,
    )
    fid_score = compute_fid(sampler_outputs, reference_dataset)
    return {
        "rollout_id": rollout_id,
        "num_samples": len(request_batch.get("prompts", [])),
        "mean_reward": fid_score,
        "fid": fid_score,
    }
```

```yaml
eval_function_dotpath: "my_project.eval.fid_evaluation"
```

## Available objects inside hooks

### `services: RolloutServices`

The service facade provides these methods:

| Method | Purpose |
|---|---|
| `services.load_prompt_batch()` | Fetch one batch of prompts from data source |
| `services.build_request(batch=..., samples_per_prompt=...)` | Build a `RolloutRequest` from prompt batch |
| `services.plan_request_batches(batch=..., samples_per_prompt=...)` | Plan request sub-batches (for direct-sampling mode) |
| `services.execute_sampling_request(request=..., sde_indices=..., ...)` | Run distributed sampling (sync) |
| `services.launch_sampling_request(request=..., ...)` | Launch sampling (async, returns future) |
| `services.resolve_launched_sampling_request(launched_request=...)` | Resolve async sampling future |
| `services.compute_advantages(rewards=..., group_ids=..., ...)` | Compute advantages via algorithm |

Useful attributes:

| Attribute | Type | Description |
|---|---|---|
| `services.samples_per_prompt` | `int` | K samples generated per prompt |
| `services.sampling_requirements` | `SamplingRequirements` | Algorithm-declared sampling needs |
| `services.sampler_validation_config` | `dict` | Sampler output validation settings |
| `services.prompt_batch_size` | `int` | Number of prompts per rollout |
| `services.data_source` | data source instance | Underlying data source |
| `services.evaluation_settings` | config object | Evaluation config (eval_batch_size, etc.) |

### `context: RolloutContext`

| Field | Type | Description |
|---|---|---|
| `context.rollout_id` | `int` | Current rollout iteration |
| `context.sde_indices` | `Optional[Set[int]]` | Which denoising steps use SDE |
| `context.collect_media_preview` | `bool` | Whether to keep decoded images for WandB |
| `context.media_max_items` | `int` | Max images for WandB preview |
| `context.debug_trace` | `Optional[Dict]` | Debug collection dict (None if not debugging) |

### `request: RolloutRequest` (in reward hook)

| Field | Access | Description |
|---|---|---|
| `request.prompts` | `List[str]` | Sample-expanded prompt list |
| `request.num_inference_steps` | `int` | Denoising steps |
| `request.height/width/num_frames` | `int` | Output geometry |
| `request.meta.get("prompt_ids")` | `Optional[List[str]]` | Per-sample prompt IDs |
| `request.meta.get("group_ids")` | `Optional[List[str]]` | Per-sample group IDs |
| `request.sampling.get("seed")` | `Optional[int]` | RNG seed |

### `samples: List[RolloutSamples]` (in reward hook)

Each `RolloutSamples` has:

| Field | Access | Description |
|---|---|---|
| `sample.latents` | `Tensor [B, C, H, W]` | Generated latents |
| `sample.timesteps` | `Tensor [T+1]` | Sigma schedule |
| `sample.aux.get("decoded_images")` | `Optional[List[PIL.Image]]` | Decoded images (dropped by default after actor-side scoring; only present when `collect_media_preview=True`) |
| `sample.aux.get("trajectories")` | `Optional[Tensor]` | Full denoising trajectory |
| `sample.aux.get("log_probs")` | `Optional[LogProbData]` | Per-step log probabilities |
| `sample.batch_size` | `int` | Number of samples in this shard |

### Return types

**`RolloutFunctionResult`** (from rollout function):
```python
RolloutFunctionResult(
    request=request,                    # The resolved RolloutRequest
    sampler_outputs=sampler_outputs,    # List[RolloutSamples] from actors
    rewards=rewards_tensor,             # [B] scalar rewards
    component_rewards={"clip": [...], "aesthetic": [...]},  # optional breakdown
    metadata={"wandb_media_preview": ...},  # optional metadata
)
```

**`RewardHookResult`** (from reward hook):
```python
RewardHookResult(
    rewards=torch.tensor([0.8, 0.6, 0.9, ...]),
    component_rewards={"my_metric": [0.8, 0.6, 0.9, ...]},  # optional
)
```

## Async training note

`train_async.py` currently requires the default `rollout_function_dotpath`. Custom rollout
functions are only supported in synchronous training (`train.py`). Custom `eval_function_dotpath`
and `reward_hook_dotpath` work in both sync and async modes.

This is because the async pipeline splits sampling into launch/resolve phases for
overlap with training, which requires direct access to the default pipeline's building
blocks (`prepare_default_rollout_plan`, `launch_request_batches_async`,
`finalize_default_rollout`).
