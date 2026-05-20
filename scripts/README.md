# Training Scripts

These shell scripts are the primary researcher entry points. Each is a
thin wrapper around `python -m diffusionrl.train_new +experiment=<name>`
that sets recipe-specific env vars (paths, output dir) and starts a Ray
cluster.

## Available launchers

| Script | Recipe (experiment) | Layout |
|---|---|---|
| `run_sd3_colocate.sh` | `flowgrpo_fast_sd3_new_colocate` | 1×8 H20, 8 vllm-omni rollout + 8 FSDP train, colocate, tensor-IPC sync |
| `run_sd3_colocate_reproduce.sh` | `flowgrpo_fast_sd3_new_colocate_reproduce` | Reproduce-grade single-node SD3 colocate |
| `run_wan21_t2v_14b_smoke.sh` | `grpo_wan21_t2v_new` | 1×8, WAN21 T2V 14B, trainside direct-sampling smoke |
| `run_hunyuan_video15_t2v_smoke.sh` | `grpo_hunyuan_video15_t2v_videopickscore_new` | 1×8, HunyuanVideo-1.5, trainside smoke |

Usage:

```bash
# Default settings
bash scripts/run_sd3_colocate.sh

# Append any Hydra overrides
bash scripts/run_sd3_colocate.sh run.num_rollouts=100

# Dry-run (print the resolved python command without executing)
DRY_RUN=1 bash scripts/run_sd3_colocate.sh
```

Each launcher accepts arbitrary trailing Hydra overrides (`run.<key>=<val>`,
`training.plan.local_batch_size=24`, …) which are forwarded to `train_new.py`.

## Helpers (not user-facing)

| File | Purpose |
|---|---|
| `_check_wandb.sh` | wandb credential preflight |
| `_mooncake.sh` | TransferQueue Mooncake storage-server lifecycle (sourced by recipes that use TransferQueue sync) |

## Override precedence

For any cfg field: **CLI override (`train_new +experiment=… key=value`) > launcher env var > experiment YAML default**.

## What goes where

| Layer | What lives there |
|---|---|
| `conf/experiment/<exp>.yaml` (hardcoded) | All recipe-defining values: model, reward, algorithm hyperparams, batch geometry (`training.plan.*`, `training.topology.actor_count`), rollout engine settings, NFT placement |
| YAML `${oc.env:VAR,default}` interpolations | Genuinely env-dependent values only: filesystem paths (`run.data_path`, `resume.output_dir`, `sync.dir`), wandb identity (`logging.run_name`, `logging.entity`) |
| Launcher env exports (`scripts/run_*.sh`) | Per-launcher defaults for those env vars (paths under `${REPO_ROOT}`, output dir, etc.) |

To override placement / batch geometry for a different cluster shape, pass
them on the CLI:

```bash
bash scripts/run_sd3_colocate.sh \
    training.topology.actor_count=32 \
    training.plan.local_batch_size=24 \
    training.plan.local_mini_batch_size=12
```

`validate_training_batch_geometry` (in `diffusionrl/config/validation.py`)
surfaces mismatches with clear error messages.

## Engine selection (Hydra)

| Mode | `rollout/engine` | `sync` |
|---|---|---|
| Direct sampling (train actor samples in-process) | `trainside` | not set (forbidden) |
| Dedicated rollout, SGLang | `sglang_new` | required (`tensor_payload`, `nccl_broadcast`, or `checkpoint_path`) |
| Dedicated rollout, vLLM-Omni | `vllm_omni` | required (same options as sglang) |

The cross-component validator enforces this biconditional.

## Local model + data paths

Recipes read inputs from defaults rooted at `${REPO_ROOT}`:

- Toy prompts: `data/samples/prompts_toy.json`, `data/samples/video_prompts_toy.txt`
- Local model mount: `models/local/<model>` (symlinks to shared model checkpoints)
- Outputs: `outputs/`

Override via env vars when invoking a launcher:

```bash
DATA_PATH=/path/to/prompts.json \
PRETRAINED_MODEL=/path/to/model \
OUTPUT_DIR=/path/to/output \
bash scripts/run_sd3_colocate.sh
```
