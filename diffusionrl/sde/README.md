# SDE Runtime

`diffusionrl.sde` owns the per-step diffusion math shared by rollout and
training:

- **σ schedule** — what noise level each inference index sees;
- **step kernels** — given a denoiser output, advance one step *and* (for SDE
  strategies) return the per-step log-probability that policy-gradient losses
  consume during replay;
- **initial noise** — deterministic per-rollout, per-group `x_T` generation
  so rollout and train-side replay sample matching trajectories.

This package does **not** decide which inference indices get SDE treatment.
That choice lives in `diffusionrl.algorithms.rollout_control` (driver-side
controller) backed by `diffusionrl.utils.scheduler_utils` (`AllSDEScheduler`,
`WindowScheduler`). Per-step math here; "which steps?" there.

## Key Files

| File | Owns |
|---|---|
| `kernels.py` | `StepStrategy` base, `SDEStrategy` mixin, and four concrete kernels: `FlowSDEStrategy`, `DanceSDEStrategy`, `CPSSDEStrategy`, `DPM2Strategy`. Each `step(...)` returns `(prev_sample, prev_sample_mean, std_var)`; SDE variants also expose log-prob computation. |
| `runtime.py` | `FlowMatchSchedulePolicy` (model-owned static **+** dynamic schedule config, incl. the `compute_mu` / `compute_sigma` methods — see the "σ Schedule Policy" section below), `get_sigma_schedule`, `ensure_req_sigmas` (pins σ onto each outbound `RolloutReq`). |
| `noise.py` | Deterministic per-group seed derivation + `generate_shared_noise` / `regen_initial_noise` (the driver-authored x_T recipe each engine regenerates) and the `generate_latents` engine-RNG fallback. |
| `rules.py` | String-level `sde_type` normalization at engine wire boundaries (e.g. SGLang kwargs). Typed Python paths read `cfg.sampling.sde_strategy` directly. |

## Registered Strategies

Step strategies register under Hydra group `sampling/sde_strategy`:

| Recipe slug | Class | Used by |
|---|---|---|
| `flow` | `FlowSDEStrategy` | FlowGRPO, default for SD3 / Qwen-Image / WAN-style flow-matching recipes |
| `dance` | `DanceSDEStrategy` | DanceGRPO |
| `cps` | `CPSSDEStrategy` | CPS-style updates |
| `dpm2` | `DPM2Strategy` | Deterministic ODE solver — evaluation only; no log-prob, so no GRPO-style training |

Pick one with:

```yaml
defaults:
  - override /sampling/sde_strategy: flow   # or dance / cps / dpm2
```

`MixGRPO` does **not** introduce a new strategy. It composes `flow` with a
windowed index schedule selected through `algorithm.scheduler.timestep_strategy: window`.

## How a Rollout Step Uses a Strategy

Each rollout engine (and train-side replay) builds the strategy once and
calls it inside the inference loop. The shape is the same across backends:

```text
sigmas   = policy.compute_sigma(num_inference_steps=..., height=..., width=...)
strategy = build(cfg.sampling.sde_strategy)
sde_idx  = control.resolve_rollout_sde_indices(current_rollout_step)

for i in range(num_inference_steps):
    eps = model.predict_noise(x_t, sigmas[i])
    if i in sde_idx:
        x_t, log_prob_i, mean_i = strategy.denoise(
            eps, x_t, sigma=sigmas[i], sigma_next=sigmas[i+1], eta=...
        )
        segment.sde_indices.append(i)
        segment.sde_logp[i] = log_prob_i
    else:
        x_t, _, _ = strategy.step(eps, x_t, sigmas[i], sigmas[i+1])
```

Two-mode contract:

- **SDE indices** record sample / mean / std and a per-step log-prob — this is
  what `DiffusionGRPO` replays during training to form the importance ratio.
- **Non-SDE indices** are pure inference; they produce no log-prob and never
  contribute gradient.

`DPM2Strategy` is deterministic at every index. It is fine for sampling-only
evaluation but cannot be the trained strategy for any GRPO recipe.

## σ Schedule Policy

`FlowMatchSchedulePolicy` is loaded from the model checkpoint once at startup
(`from_pretrained` reads the `scheduler/transformer/vae` JSONs) and shared
across rollout and train actors. It carries:

- `shift` — **static** FlowMatch time-shift scalar; used only on the static
  branch (`use_dynamic_shifting=False`);
- `use_dynamic_shifting` — selects the static vs dynamic branch;
- `base_shift`, `max_shift`, `base_image_seq_len`, `max_image_seq_len`,
  `time_shift_type` — the dynamic-shift block;
- `vae_scale_factor`, `patch_size` — derive `image_seq_len` from
  `height x width`.

On the dynamic branch the per-request μ is computed by
`policy.compute_mu(image_seq_len, num_inference_steps)` — **the single
per-model override point**. The default delegates to `calculate_dynamic_mu`
(linear in `image_seq_len`); a model whose μ differs subclasses the policy and
overrides `compute_mu` (e.g. FLUX.2-klein's empirical μ, which also depends on
`num_inference_steps`). The schedule *application* (base grid + diffusers
time-shift) stays shared — only the μ value is model-specific.

`ensure_req_sigmas(req, policy)` pins σ onto each outbound `RolloutReq` so
every rollout actor (including dedicated backends that do not re-read the
checkpoint) samples on the same schedule the trainer will replay.

## Initial Noise

`noise.py` produces deterministic per-rollout x_T. The driver
(`DiffusionTrainer._build_req`) does NOT materialize the tensor — it authors a
small **recipe** on the `RolloutReq`: per-sample `init_noise_group_ids` (each
keyed on `rollout_id` + the stable sample/group id, e.g.
`f"r{rollout_id}:{sample_id}"`) plus `init_noise_latent_shape` (the pipeline's
own `latent_shape`). Each engine regenerates the byte-identical x_T from it:

```text
x_T = regen_initial_noise(
    noise_group_ids=req.init_noise_group_ids,   # per-sample, rollout-keyed
    base_seed=sampling_params.seed,             # raw config seed (no rollout mix)
    latent_shape=req.init_noise_latent_shape,   # = pipeline.latent_shape(...)
)   # drawn on CPU-fp32, then moved/cast to the engine device
```

- Per-rollout variety comes from `rollout_id` *inside the group-id string*
  (not a seed mix). CPU randn is bit-stable across machines for a fixed torch
  version, so trainside / sglang / vllm-omni agree on x_T to the byte.
- All samples in one GRPO group share noise if `sampling.init_same_noise=True`
  (group-keyed ids); otherwise each sample draws an independent slice
  (stable-sample-id-keyed, so a sample keeps its x_T under resume / re-shard).
- `request_conditions["initial_latents"]` still takes precedence when present
  (img2img / first-frame conditioning); the recipe only fills the t2i x_T.

## What Lives Elsewhere

| Concern | Owner |
|---|---|
| Which inference indices run SDE on a given rollout step | `diffusionrl.algorithms.rollout_control.GRPORolloutControl.resolve_rollout_sde_indices(...)` |
| Window / progressive / random index scheduling (MixGRPO) | `diffusionrl.utils.scheduler_utils.WindowScheduler` |
| Mapping `algorithm.scheduler.*` YAML into a scheduler object | `diffusionrl.utils.scheduler_utils.create_indices_scheduler` |
| Per-model σ override (e.g. video models with custom shift) | `diffusionrl/models/<name>/diffusion.py` |
| Filtering trained steps after rollout (`skip_last_timestep`, `skip_initial_timesteps`) | `GRPORolloutControl.get_filtered_training_indices(...)` |

See `diffusionrl/algorithms/README.md` for the algorithm-side view of this
boundary.
