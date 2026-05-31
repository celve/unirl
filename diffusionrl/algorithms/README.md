# Algorithms Package

`diffusionrl.algorithms` has two separate surfaces:

- `cfg.algorithm`: driver-side rollout control. It decides rollout batch size,
  SDE step selection, and reward-to-advantage normalization.
- `cfg.algorithms.<slot>`: train-side loss objects. They consume
  `RolloutResp.tracks[slot]` and call `backward()`.

Keep this split in mind when reading recipes. Most "algorithm" confusion comes
from mixing these two layers.

## Algorithm Families

| Recipe family | Driver side | Train side | What changes |
|---|---|---|---|
| GRPO | `algorithm: grpo` | `DiffusionGRPO` / `ARGRPO` | Group-normalized advantages + PPO-style clipped ratio loss |
| FlowGRPO | `algorithm: grpo` | `DiffusionGRPO` | GRPO with `sampling/sde_strategy: flow` |
| DanceGRPO | `algorithm: grpo` | `DiffusionGRPO` | GRPO with `sampling/sde_strategy: dance` |
| MixGRPO | `algorithm: grpo` | `DiffusionGRPO` | GRPO with windowed SDE-index scheduling |
| NFT | `algorithm: nft` | `DiffusionNFT` | Forward-process loss with default/old LoRA adapters |

FlowGRPO, DanceGRPO, and MixGRPO are recipe compositions around the same
train-side GRPO class. Add a new train-side class only when the loss math
changes.

## Computation Flow

One rollout-train iteration looks like this:

```text
build(cfg.algorithm)
  -> RolloutPipeline.plan_requests(...)
       chooses SDE indices for this rollout
  -> RolloutActorGroup.run_rollout_pipeline(...)
       generates RolloutResp, attaches rewards, computes advantages
  -> TrainActorGroup.train(...)
       shards RolloutResp across train actors
  -> StageTrainStack.train_optimizer_step(...)
       dispatches each tracks[slot] to cfg.algorithms.<slot>
  -> StageAlgorithm.compute_loss_and_backward(...)
       replays the stage, computes loss, calls backward
```

`RolloutResp` is the important boundary; each `tracks[slot]` carries:

- `conditions`: typed prompt/media conditions for train-side replay;
- `segment`: diffusion or AR traces emitted by rollout;
- `rewards`: scalar reward per sample;
- `advantages`: normalized reward signal used by train-side losses.

Slot names must match:

```text
cfg.algorithms.image -> resp.tracks["image"]
cfg.algorithms.ar    -> resp.tracks["ar"]
```

Missing slots raise by default so a broken rollout cannot silently become a
no-gradient training step.

## Reward to Advantage

A reward turns into a gradient signal across four places in the code. Grep
the function name on the right to read the actual implementation:

```text
[1] reward components on the rollout actor
    diffusionrl.reward.service.RewardService.score_and_attach
      -> resp.rewards : Tensor[B]
         resp.component_rewards   (per-component, for logging)

[2] fused pipeline aggregates one shard's responses
    the rollout pipeline's run_rollout_pipeline
      -> all_rewards   = torch.cat([r.rewards for r in responses])
         all_group_ids = [gid for r in responses for gid in r.group_ids]

[3] rollout control z-scores rewards into advantages
    diffusionrl.algorithms.rollout_control.GRPORolloutControl.compute_advantages
      -> resp.advantages : Tensor[B]   (scope = group | global)

[4] train actor expands advantages and applies the clipped ratio
    diffusionrl.algorithms.grpo.DiffusionGRPO.compute_loss_and_backward
      -> adv_b = advantages.detach().reshape(-1, 1).expand_as(new_logp)
         loss  = _grpo_clip_loss(new_logp, old_logp, adv_b, clip_range)
```

The knobs that control each step:

| Config | Effect |
|---|---|
| `reward.backend` | the reward backend — a local scorer (PickScore, HPS, OCR, …) or the remote RewardService client |
| `algorithm.adv_normalization_scope` | `group` normalizes within each prompt group; `global` normalizes across the batch |
| `algorithm.samples_per_prompt` | expected group size for grouped normalization |
| `algorithm.use_global_std` | grouped normalization may share one global std |
| `algorithms.<slot>.clip_range` | PPO clip ε passed to `_grpo_clip_loss` |

GRPO uses `advantages` as the multiplier in the clipped ratio objective. NFT
also receives `advantages`, but maps clipped advantages into `r in [0, 1]` for
its positive/negative reconstruction loss (see `DiffusionNFT.compute_loss_and_backward`).

## GRPO Loss

`DiffusionGRPO` trains a diffusion `LatentSegment`.

For one micro-batch:

1. choose trainable SDE step ids from `segment.sde_indices`;
2. replay the stage at current weights to compute `new_logp`;
3. gather rollout-time `old_logp` from `segment.sde_logp`;
4. expand per-sample advantages from `[B]` to `[B, num_steps]`;
5. apply PPO-style clipping:

```text
ratio = exp(new_logp - old_logp)
loss = mean(max(
  -adv * ratio,
  -adv * clamp(ratio, 1 - clip_range, 1 + clip_range)
))
```

`ARGRPO` uses the same clipped objective for text traces. The only shape
difference is that sample-level advantages are repeated over each sample's
token span before computing the per-token loss.

Some rollout engines emit old log-probs directly. SGLang replay-mode rollouts
may emit trajectory data without `segment.sde_logp`; in that case
`DiffusionGRPO.prepare_segment(...)` fills old log-probs once with a
`torch.no_grad()` replay before the multi-update loop. This keeps the old
policy fixed across `training.plan.num_updates_per_batch`.

## NFT Loss

`DiffusionNFT` is forward-process training. It does not use rollout SDE
log-probs. It trains from the clean final latent and a set of training
timesteps.

For one micro-batch:

1. read `x0 = segment.latents[:, -1]`;
2. resolve training timesteps from `segment.sigmas` or random sampling;
3. clip `advantages` and map them to `r in [0, 1]`;
4. for each timestep `t`, construct:

```text
xt = (1 - t) * x0 + t * noise
```

5. predict noise with the trainable `default` adapter;
6. predict noise with the EMA-tracked `old` adapter via
   `NFTLoRAPolicy.with_old_adapter()`;
7. form positive and negative predictions;
8. reconstruct `x0` from both and compute:

```text
loss = mean((r * pos_loss + (1 - r) * neg_loss) / beta) * adv_clip_max
```

The actual implementation optionally normalizes the MSE terms with adaptive
per-sample weights. The `old` adapter used inside the NFT loss is maintained
by `NFTLoRAPolicy`; this is a separate concept from the rollout-time EMA
wrapping described in the next section.

## EMA-Rollout (Off-Policy)

`StageAlgorithm.requires_ema_rollout` is a class attribute that opts an
algorithm into sampling rollouts with EMA-smoothed weights instead of the
trainable weights. `DiffusionGRPO` leaves it `False`; `DiffusionNFT` sets it
`True`. The flag wires through four call sites:

```text
class attr   StageAlgorithm.requires_ema_rollout         (diffusionrl/algorithms/base.py)
             DiffusionNFT.requires_ema_rollout = True     (diffusionrl/algorithms/nft.py)

driver read  _should_use_ema_rollout(cfg)                 (diffusionrl/train.py)
               walks cfg.algorithms.<slot>, resolves each _target_,
               returns True if any slot's class declares the flag.

gate         _use_ema_rollout = _should_use_ema_rollout(cfg) if direct_sampling else False

rollout wrap with train_group.use_eval_ema() if _use_ema_rollout else nullcontext():
                 rollout_pipeline.generate(...)
             (train-group eval-EMA scope)

shadow swap  TrainActor.apply_eval_ema  ─►  EMAPolicy.apply_ema_to_model
             TrainActor.restore_from_eval ─► EMAPolicy.restore_from_ema
                                                          (diffusionrl/training/ema_policy.py)
```

Three gotchas:

- **GRPO must NOT set this flag.** On-policy rollout requires
  `exp(new_logp - old_logp) == 1` on the first update, which only holds when
  rollout and replay use identical weights. Setting `requires_ema_rollout=True`
  on a clipped-ratio algorithm silently injects a bias on update #1.
- **The flag only fires in direct sampling.** Dedicated rollout engines
  maintain their own weight copy; for EMA-style sampling there, ship EMA
  weights through a `sync:` backend (see
  `diffusionrl/distributed/weight_sync/README.md`), not through this attribute.
- **`use_eval_ema()` is a no-op without `EMAPolicy`.** `TrainActor.apply_eval_ema`
  walks the policy chain and returns silently when no `EMAPolicy` is found;
  if the experiment YAML omits `EMAPolicy` under `training.policies`, EMA
  wrapping never happens and no error is raised.

## SDE Boundary

Algorithms decide **which inference steps** use SDE; the per-step math lives
in `diffusionrl/sde/`. When something does not add up, jump straight to the
right file:

| Question | Read |
|---|---|
| Which indices run SDE on rollout step *k*? | `GRPORolloutControl.resolve_rollout_sde_indices` |
| Which indices contribute to training loss after rollout? | `GRPORolloutControl.get_filtered_training_indices` |
| What is the per-step kernel that produces `log_prob_i`? | `diffusionrl/sde/kernels.py` (`FlowSDEStrategy`, `DanceSDEStrategy`, `CPSSDEStrategy`, `DPM2Strategy`) |
| Where does the σ schedule come from? | `diffusionrl/sde/runtime.py` (`FlowMatchSchedulePolicy.compute_sigma`, pinned onto the request by `ensure_req_sigmas`) |
| What is the MixGRPO sliding-window schedule? | `diffusionrl.utils.scheduler_utils.WindowScheduler` (selected by `algorithm.scheduler.timestep_strategy: window`) |

`NFTRolloutControl.resolve_rollout_sde_indices` returns `None` because NFT
does not train on rollout log-probs. `MixGRPO` is not a separate SDE kernel;
it is GRPO with a windowed SDE-index scheduler.

## YAML Shape

Minimal GRPO pattern:

```yaml
defaults:
  - override /algorithm: grpo
  - override /sampling/sde_strategy: flow

algorithm:
  prompts_per_rollout: 4
  samples_per_prompt: 8
  scheduler:
    timestep_fraction: [0.0, 0.5]
    num_sde_steps: 3

algorithms:
  image:
    _target_: diffusionrl.algorithms.DiffusionGRPO
    stage_attr: diffusion
    conditions_cls: diffusionrl.models.sd3.conditions.SD3Conditions
    clip_range: 1.0e-4
    params:
      _target_: diffusionrl.models.sd3.diffusion.SD3DiffusionParams
      num_inference_steps: ${sampling.num_inference_steps}
      guidance_scale: ${sampling.guidance_scale}
```

Minimal NFT pattern:

```yaml
defaults:
  - override /algorithm: nft

algorithm:
  scheduler:
    num_sde_steps: 0

algorithms:
  image:
    _target_: diffusionrl.algorithms.DiffusionNFT
    stage_attr: diffusion
    conditions_cls: diffusionrl.models.sd3.conditions.SD3Conditions
    beta: 1.0
    adv_clip_max: 5.0
    train_timestep_mode: all
```

If the selected pipeline does not expose `<slot>_params`, provide an inline
`params:` block so the algorithm can replay the stage during training.

## Adding an Algorithm

1. Decide whether the change is rollout control, train-side loss, or both.
2. For train-side loss, subclass `StageAlgorithm`.
3. Implement `compute_loss_and_backward(...)` and return `AlgorithmStepResult`.
4. Override `prepare_segment(...)` only when old/reference fields need
   pre-update materialization.
5. Set `requires_ema_rollout = True` only for off-policy algorithms that need
   EMA rollout sampling.
6. Wire the class under `algorithms.<slot>` in an experiment YAML.
