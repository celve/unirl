# Algorithms Package

`unirl.algorithms` provides the **train-side loss algorithms**. Each is a
`StageAlgorithm` subclass that consumes a rollout track (`RolloutResp.tracks[name]`),
replays the stage at current weights, computes a loss, and calls `backward()`.

A recipe binds **one** algorithm per track. For a single-track recipe that is the
top-level `cfg.algorithm` node; for a multi-track recipe (e.g. PE) each track
carries its own node — `diffusion.algorithm` and `ar.algorithm`. The
`TrainStack` (`unirl.train.stack.TrainStack`) holds exactly one algorithm and
runs the mini-batch optimizer loop; multi-track training uses sibling
`TrainStack`s, not a slot dispatcher.

There is no separate "driver-side rollout control" object. Reward→advantage
shaping and SDE-index selection live on typed objects (`RolloutTrack` and
`DiffusionSamplingParams`), described below.

## Algorithm Families

| Algorithm | Class (`_target_`) | Domain | What it does |
|---|---|---|---|
| GRPO | `unirl.algorithms.diffusion_grpo.DiffusionGRPO`, `unirl.algorithms.ar_grpo.ARGRPO` | diffusion / AR | Group-normalized advantages + PPO-style clipped-ratio loss |
| DanceGRPO / MixGRPO | `DiffusionGRPO` | diffusion | GRPO with an `sde_strategy` / windowed-scheduler swap (recipe composition, same loss class) |
| Flow-DPPO | `unirl.algorithms.dppo.DiffusionDPPO` | diffusion | KL-divergence-masked DPPO objective |
| NFT | `unirl.algorithms.nft.DiffusionNFT` | diffusion | Forward-process loss with dual `default`/`old` (EMA) adapters |
| SPO-DPPO | `unirl.algorithms.spo_dppo.ARSPODPPO` | AR | TV/KL trust-region objective for text traces |

DanceGRPO and MixGRPO are recipe compositions around the same `DiffusionGRPO`
class (a different `sampling.sde_strategy` or `algorithm/scheduler`). Add a new
class only when the loss math changes.

## Computation Flow

One rollout–train iteration, as driven by the trainer (e.g. `trainer/diffusion.py`):

```text
rollout engine
  -> RolloutResp.tracks[name]                      (conditions, segment, rewards)
RewardService.score_and_attach(req=, track=)        -> track.rewards
RolloutTrack.compute_advantages(scope=...)          -> track.advantages
TrainStack.train_track(track)
  -> prepare_segment(...)                           (materialize old/reference fields once)
  -> mini-batch loop over num_updates_per_batch
       StageAlgorithm.compute_loss_and_backward(...) (replay stage, loss, backward)
  -> optimizer_step
```

Each `RolloutResp.tracks[name]` carries:

- `conditions`: typed prompt/media conditions for train-side replay;
- `segment`: diffusion or AR traces emitted by rollout;
- `rewards`: scalar reward per sample (plus `component_rewards` for logging);
- `advantages`: normalized reward signal used by the loss.

The track key is a **stage name** (e.g. `"diffusion"`, `"ar"`), and a single
`TrainStack` binds exactly one track.

## Reward to Advantage

A reward becomes a gradient signal in three steps — grep the symbol on the right
to read the implementation:

```text
[1] reward backend scores the rollout
    unirl.reward.service.RewardService.score_and_attach   (unirl/reward/service.py)
      -> track.rewards            : Tensor[B]
         track.component_rewards  (per-component, for logging)

[2] z-score rewards into advantages
    RolloutTrack.compute_advantages(normalize=, eps=, scope=)   (unirl/types/rollout_resp.py)
      -> track.advantages : Tensor[B]   (scope = group | global)

[3] the loss expands advantages and applies its objective
    DiffusionGRPO.compute_loss_and_backward                (unirl/algorithms/diffusion_grpo.py)
      -> adv_b = advantages.detach().reshape(-1, 1).expand_as(new_logp)
         loss  = _grpo_clip_loss(new_logp, old_logp, adv_b, clip_range)
```

The knobs that control each step:

| Config | Effect |
|---|---|
| `reward.backend` | the reward backend — a local scorer (PickScore, HPS, OCR, …) or the remote HTTP client |
| `algorithm` advantage `scope` | `group` normalizes within each prompt group; `global` normalizes across the batch (the live diffusion path honors `scope` only) |
| `algorithm.clip_range` | PPO clip ε passed to `_grpo_clip_loss` (GRPO / DPPO) |

GRPO uses `advantages` as the multiplier in the clipped-ratio objective. NFT
also receives `advantages`, but clips them and maps to `r ∈ [0, 1]` for its
positive/negative reconstruction loss (see `DiffusionNFT.compute_loss_and_backward`).

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

`ARGRPO` uses the same clipped objective for text traces; sample-level
advantages are repeated over each sample's token span before the per-token loss
(`_expand_advantages_to_tokens`).

Some rollout engines emit old log-probs directly. SGLang replay-mode rollouts
may emit trajectory data without `segment.sde_logp`; in that case
`DiffusionGRPO.prepare_segment(...)` fills old log-probs once with a
`torch.no_grad()` replay before the multi-update loop, keeping π_old fixed
across `stack.num_updates_per_batch`.

## NFT Loss

`DiffusionNFT` is forward-process training. It does not use rollout SDE
log-probs; it trains from the clean final latent and a set of training
timesteps.

For one micro-batch:

1. read `x0 = segment.latents[:, -1]`;
2. resolve training timesteps from `segment.sigmas` or random sampling;
3. clip `advantages` and map them to `r ∈ [0, 1]`;
4. for each timestep `t`, construct `xt = (1 - t) * x0 + t * noise`;
5. predict noise with the trainable `default` adapter;
6. predict noise with the EMA-tracked `old` adapter via `nft_lora_policy.use_shadow()`;
7. form positive and negative predictions;
8. reconstruct `x0` from both and compute:

```text
loss = mean((r * pos_loss + (1 - r) * neg_loss) / beta) * adv_clip_max
```

The implementation optionally normalizes the MSE terms with adaptive per-sample
weights (`use_adaptive_weight`). The `old` adapter is the train backend's EMA
shadow (`backend.ema`); the constructor raises if it lacks `use_shadow`, so a
mis-wired NFT recipe fails loud rather than silently skipping EMA.

## EMA-Rollout (Off-Policy)

NFT samples rollouts from **EMA-smoothed** weights rather than the trainable
weights. The trainer gates this by the algorithm being `DiffusionNFT`
(`trainer/diffusion.py` sets `_uses_ema = algo_target.endswith("DiffusionNFT")`);
during rollout it wraps the model with `backend.apply_eval_ema()` /
`restore_from_eval()`, which swap the EMA shadow in via `EMA.use_shadow()`
(`unirl/train/ema.py`).

Notes:

- **On-policy GRPO / DPPO must sample with the trainable weights.** The clipped
  ratio needs `exp(new_logp - old_logp) == 1` on update #1, which only holds when
  rollout and replay use identical weights — so only NFT opts into EMA rollout.
- **In dedicated-rollout modes** the rollout engine holds its own weight copy;
  EMA-style sampling there means shipping EMA weights through a `sync:` backend
  (see `unirl/distributed/weight_sync/README.md`), not through the trainer wrap.
- `StageAlgorithm.requires_ema_rollout` still exists as a class attribute but is
  currently informational; the live gate is the `DiffusionNFT` target check above.

## SDE Boundary

Algorithms decide **which inference steps** use SDE; the per-step math lives in
`unirl/sde/`. When something does not add up, jump straight to the right file:

| Question | Read |
|---|---|
| Which indices run SDE / contribute to training loss? | `unirl.types.sampling.DiffusionSamplingParams.resolve_sde_indices` (called by the trainer) |
| What is the per-step kernel that produces `log_prob_i`? | `unirl/sde/kernels.py` (`FlowSDEStrategy`, `DanceSDEStrategy`, `CPSSDEStrategy`, `DPM2Strategy`) |
| Where does the σ schedule come from? | `unirl/sde/runtime.py` (`FlowMatchSchedulePolicy`, pinned onto the request by `ensure_req_sigmas`) |
| What is the MixGRPO sliding-window schedule? | `unirl.utils.scheduler_utils.WindowScheduler` (the `algorithm/scheduler` group, `timestep_strategy: window`) |

NFT does not train on rollout log-probs, so it has no rollout-SDE-index
selection. `MixGRPO` is not a separate SDE kernel; it is GRPO with a windowed
SDE-index scheduler.

## Multi-Update

`stack.num_updates_per_batch` partitions one rollout shard into N disjoint
mini-batches (one optimizer step each), with π_old frozen once by
`prepare_segment`. This is gated on `StageAlgorithm.supports_multi_update`:
`DiffusionGRPO` / `DiffusionDPPO` support it; `ARGRPO` / `ARSPODPPO` are
single-update-only and `TrainStack` raises if a recipe sets `> 1` for them.

## YAML Shape

GRPO (single track) — the loss is the `algorithm:` node; the stack carries the
mini-batch geometry:

```yaml
algorithm:
  _target_: unirl.algorithms.diffusion_grpo.DiffusionGRPO
  stage_attr: diffusion
  clip_range: 1.0e-4
  conditions_cls:
    _target_: hydra.utils.get_class
    path: unirl.models.sd3.conditions.SD3Conditions
  params: ${sampling}        # replay uses the same guidance_scale / eta as rollout

stack:
  _target_: unirl.train.stack.TrainStack
  micro_batch_size: 1
  max_grad_norm: 1.0
  num_updates_per_batch: 2   # PPO mini-batches per rollout (π_old frozen once)
```

NFT:

```yaml
algorithm:
  _target_: unirl.algorithms.nft.DiffusionNFT
  stage_attr: diffusion
  conditions_cls:
    _target_: hydra.utils.get_class
    path: unirl.models.sd3.conditions.SD3Conditions
  params: ${sampling}
  beta: 1.0
  adv_clip_max: 5.0
  use_adaptive_weight: true
  train_timestep_mode: all
```

Multi-track recipes (e.g. PE) nest one `algorithm:` node per track, under the
track name (`diffusion:` and `ar:`), each with its own `_target_`.

## Adding an Algorithm

1. Subclass `StageAlgorithm` (`unirl/algorithms/base.py`).
2. Define a config with `@register_config(...)` subclassing `BaseAlgorithmConfig`.
3. Implement `compute_loss_and_backward(...)` (replay the stage, compute loss, `backward()`).
4. Override `prepare_segment(...)` only when old/reference fields need
   pre-update materialization (e.g. freezing π_old for multi-update).
5. Set `supports_multi_update` to match whether the loss is valid across
   multiple optimizer steps on one rollout shard.
6. Bind the class under the track's `algorithm:` node in a recipe.
