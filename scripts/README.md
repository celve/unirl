# Training Scripts

These shell scripts are the primary researcher entrypoints in this repo.

## Multi-node entrypoint: `scripts/train.sh`

`scripts/train.sh <experiment> [role] [hydra overrides...]` is the generic
multi-node launcher. It owns the NCCL env block, TaiJi/Jizhi cluster
auto-detection (`CHIEF_IP`/`INDEX`/`HOST_NUM`/`HOST_GPU_NUM`), Ray cluster
bootstrap, and role dispatch (`auto`/`head`/`worker`/`train`/`stop`/`status`).
The launcher injects **zero Hydra overrides** — it exports env vars consumed
by YAML `${oc.env:VAR,default}` interpolations and runs
`python -m diffusionrl.train +experiment=<name> "$@"`.

Per-experiment recipes live as ~20-line wrappers under `reproduce_scripts/`
that set environment-derived values (paths, wandb run name) and `exec` into
`train.sh`.

### Override precedence

For any cfg field: **CLI override > env var > YAML default**.

### What goes where

| Layer | What lives there |
|---|---|
| `conf/experiment/<exp>.yaml` (hardcoded) | All recipe-defining values: model, reward, algorithm hyperparams, scheduler, EMA, sampling, batch geometry (`training.plan.*`, `training.topology.actor_count`), rollout engine settings, NFT placement |
| YAML `${oc.env:VAR,default}` interpolation | Genuinely env-dependent values only: cluster GPU/node count (`placement.*`), filesystem paths (`run.data_path`, `resume.output_dir`, `sync.dir`), wandb identity (`logging.run_name`, `logging.entity`), SGLang GPU split |
| Wrapper env exports (`reproduce_scripts/<exp>.sh`) | Defaults for the env vars above (paths under `${REPO_ROOT}`, run name, etc.) |
| `scripts/train.sh` (env passthrough) | Cluster topology (`NUM_NODES`, `GPUS_PER_NODE`) detected from TaiJi env; passes through wrapper-set env vars to the Python process |

Recipe values that are anchored to a specific cluster shape (typically 2-node
× 8-GPU) are hardcoded in the YAML. To run a recipe on a different cluster
shape, override the placement + batch geometry on the CLI:
```
bash reproduce_scripts/train_grpo_sd3_multinode.sh auto \
    training.topology.actor_count=32 \
    training.plan.local_batch_size=24 \
    training.plan.local_mini_batch_size=12
```
The `validate_training_batch_geometry` and `validate_rollout_batch_geometry`
validators (in `diffusionrl/config/validation.py`) surface mismatches with
clear error messages.

## Layout

`scripts/` hosts only the generic launchers (`train.sh`, `train_tq.sh`)
plus their helpers (`_check_wandb.sh`, `_mooncake.sh`). Per-recipe
wrappers live under `reproduce_scripts/` and `exec` into one of these
launchers.
A local `configs/recipes/` directory may exist in some working trees, but it
is gitignored and should not be treated as the public repo interface.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- OCR toy data: `data/samples/ocr_prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

Use any wrapper under `reproduce_scripts/` (e.g. `reproduce_scripts/train_grpo_sd3_multinode.sh`) for a smoke run; override `DATA_PATH` to a toy file from `data/samples/` if you don't want to touch real datasets.

Precision config:

- Use `precision.training.*` for training model load precision, FSDP param precision, and train-side autocast.
- Use `precision.rollout.*` for sampler/replay autocast plus trajectory/logprob storage precision.
- In dedicated SGLang rollout, prompt encoder precision also follows `precision.rollout.autocast_precision`.

Terminology:

- `rollout_id` is the outer rollout-train loop step.
- For experiment tracking it is close to a framework-level global step.
- It is not guaranteed to equal raw optimizer step count when gradient accumulation or inner epochs are used.

Eval data:

- `data_path` is the training prompt source.
- `eval_data_path` is optional; when unset, periodic eval reads a deterministic, unshuffled view of `data_path`.
- For a real validation split, set `eval_data_path` explicitly.
- Prompt datasets should provide text via `prompt` or `caption`; legacy embedding fields are no longer supported.
- Prompt datasets should ideally provide `prompt_id`; when omitted, the prompt loaders now synthesize deterministic IDs.

## Engine note

For training-actor direct sampling, set `rollout.mode=direct_sampling`
and omit `cfg.sync` entirely (it must not be set).
Leave `rollout.rollout_engine` unset in direct mode.
For SGLang, use `rollout.mode=separate` or `rollout.mode=colocate`
with `rollout.rollout_engine=sglang`, and add a dedicated-rollout
weight-sync variant via the `sync` group: `+sync=tensor_payload`,
`+sync=nccl_broadcast`, or `+sync=checkpoint_path`.
Dedicated rollout modes also require `rollout.num_gpus_per_actor` to be set explicitly.

### SGLang remote scheduler mode (TCP, non-HTTP data plane)

Use `rollout.sglang_local_mode=false` and pass explicit scheduler endpoint(s)
through `rollout.sglang_kwargs`:

```yaml
rollout:
  mode: separate
  rollout_engine: sglang
  num_gpus_per_actor: 4
  tp_size: 4
  sglang_local_mode: false
  sglang_kwargs:
    remote_scheduler_endpoints:
      - tcp://10.0.0.11:35555
      - tcp://10.0.0.12:35555
```

Notes:
- When `remote_scheduler_endpoints` is set, rollout actors map by rank (`rank % len(endpoints)`).
- Rollout weight updates are deduplicated per logical scheduler endpoint (avoids repeated updates to the same scheduler).
- SGLang rollout now encodes prompts inside `generate()` unconditionally so sampler outputs satisfy the rollout embedding contract without a driver-side fallback path.

## Local model setup

Models are loaded from `models/local/` (symlinks to `shared_models/`).
If local paths don't exist, scripts automatically fall back to HuggingFace
downloads (configured in `diffusionrl/config/arguments.py`).

Override default paths by environment variables when invoking a `reproduce_scripts/` wrapper (e.g. `DATA_PATH=/path/to/prompts.json PRETRAINED_MODEL=/path/to/model bash reproduce_scripts/<exp>.sh`).

All commands invoke canonical package entrypoint:

```bash
python -m diffusionrl.train ...
```
