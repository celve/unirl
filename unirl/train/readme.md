# Train Stack

`unirl/train` is the v2 single-stage training stack. It owns model training
state (weights, optimizer, scheduler, EMA, structural injection) and the
loss/backward sequencing for one rollout track.

Two `Remote` siblings split the work:

- **`FSDPBackend`** (`backend/fsdp.py`) owns the trainable model, optimizer,
  scheduler, EMA shadow, and structural injection.
- **`TrainStack`** (`stack.py`) owns loss/backward sequencing: it takes handles
  to one `FSDPBackend` and one `StageAlgorithm` and runs the micro-batch /
  multi-update optimizer loop. It is single-stage by design — one track, no
  track-name dict. Multi-track training (e.g. PE) uses sibling `TrainStack`s.

HunyuanImage3's mixed AR + diffusion training uses `hi3_stack.py`, a multi-stage
variant.

## Key Files

| File | Responsibility |
|---|---|
| `stack.py` | `TrainStack` (single-stage): `train_track` → `prepare_segment` → mini-batch `train` loop → `optimizer_step`; returns `TrainStepResult` |
| `backend/fsdp.py` | `FSDPBackend`: FSDP2-wraps the model, builds optim/sched/EMA, `optimizer_step` (grad-clip + step + sched + EMA, skips non-finite grads), `onload`/`offload`, `save`/`load`, `apply_eval_ema`/`restore_from_eval` |
| `backend/base.py` | `OptimizerConfig`, `LrSchedulerConfig`, `TrainTopology` |
| `inject.py` | Structural injection before the FSDP wrap: `inject_lora`, `inject_nft` (dual `default`+`old` adapters), `inject_mirror` (full-model shadow), `fsdp_wrap` |
| `ema.py` | `EMA` shadow + `make_decay_fn`; `use_shadow()` context |
| `shadow.py` | `Shadow` — swap-in/swap-out abstraction over adapter-vs-mirror EMA |
| `configs.py` | `LoraConfig`, `EmaLoraConfig`, `EmaFullConfig`, `FSDPConfig` |
| `factories.py` | `build_optimizer` (AdamW), `build_lr_scheduler` (constant / linear / cosine) |
| `fsdp_utils.py` | FSDP2 helpers (block-class discovery, mesh) |
| `hi3_stack.py` | Multi-stage train stack for HunyuanImage3 (AR + diffusion) |

## Train-Step Contract

`TrainStack.train_track(track)` is the driver entry. For one rollout track:

1. `prepare_segment(track)` — materialize old/reference fields once (e.g. freeze
   π_old for multi-update GRPO).
2. partition the shard into `num_updates_per_batch` disjoint mini-batches.
3. for each mini-batch, `train(...)`: `zero_grad` → micro-batch loop
   (`algorithm.compute_loss_and_backward`) → `FSDPBackend.optimizer_step`.
4. return a `TrainStepResult` (loss / grad-norm / metrics).

`num_updates_per_batch > 1` runs several PPO updates on one rollout shard with
π_old frozen; only multi-update-capable algorithms may set it (see
`unirl/algorithms/README.md`).

## FSDP Backend

`FSDPBackend` is constructed once and:

- dispatches structural injection (`inject_lora` / `inject_nft` / `inject_mirror`)
  then `fsdp_wrap` — FSDP2 `fully_shard` per discovered model block class, a
  mixed-precision policy (compute dtype, fp32 reduce), and optional CPU offload /
  activation checkpointing / `torch.compile` / HSDP mesh;
- builds the optimizer and LR scheduler (and EMA when configured);
- `optimizer_step` clips grads, steps, advances the scheduler, updates EMA, and
  skips the step on non-finite grads;
- `onload` / `offload` move state on/off GPU between phases (gated by
  `enable_fsdp_offload`) for colocate weight handoff;
- `save` / `load` checkpoint;
- `apply_eval_ema` / `restore_from_eval` swap the EMA shadow in for NFT rollout.

`lora_cfg` and `ema_lora_cfg` are mutually exclusive (the constructor guards this).

## Structural Injection

Injection runs in the backend constructor, before the FSDP wrap:

- `inject_lora` — a plain LoRA adapter on the trainable stage.
- `inject_nft` — dual adapters: a trainable `default` plus an `old` EMA shadow,
  used by the NFT loss; the `old` adapter is driven by an `EMA`.
- `inject_mirror` — full-model `shadow_*` parameters mirrored into an `EMA`
  (full-weight EMA rather than adapter EMA).

## EMA Shadow

`EMA` (`ema.py`) maintains a shadow that updates on `optimizer_step` or
`on_rollout_end` with a decay from `make_decay_fn` (full: `min((1+t)/(10+t),
target)`; LoRA: constant / linear / warmup). `Shadow` (`shadow.py`) abstracts
swap-in/swap-out so the same path serves adapter EMA and full-mirror EMA;
`apply_eval_ema` / `restore_from_eval` use it to sample NFT rollouts from the
EMA weights.

## How the Entrypoints Drive It

`train_diffusion` / `train_vlm` / `train_pe` build the trainer
(`unirl/trainer/*.py`), which constructs the `FSDPBackend`, the `StageAlgorithm`,
and the `TrainStack` as sibling `Remote`s inside a placement block, then calls
`train_track` each iteration. HunyuanImage3 (`train_hi3`) uses the multi-stage
`hi3_stack.py`.
