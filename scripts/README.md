# Training Scripts

`scripts/` is intentionally thin. Training semantics live in
`conf/experiment/*.yaml`; shell scripts only prepare the runtime, start Ray,
set path/logging defaults, and forward Hydra overrides.

For the experiment inventory and config ownership rules, see
`../conf/README.md`.

## Launchers

| Script | Use |
|---|---|
| `run_experiment_single_node.sh <experiment>` | Generic 1-node launcher for any `conf/experiment/<experiment>.yaml` |
| `run_experiment_multinode.sh <experiment>` | Generic role-aware launcher for multi-node jobs; workers join Ray and the head runs training |

Examples:

```bash
bash scripts/run_experiment_single_node.sh grpo_wan21_t2v
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate
bash scripts/run_experiment_multinode.sh flowgrpo_fast_qwen_image_2x8
bash scripts/run_experiment_multinode.sh nft_qwen_image_4x8 run.num_rollouts=100
DRY_RUN=1 bash scripts/run_experiment_single_node.sh mixgrpo_sd35
```

The generic launchers set these environment-derived overrides:

- `run.data_path=${DATA_PATH}`
- `run.eval_data_path=${EVAL_DATA_PATH}`
- `resume.output_dir=${OUTPUT_DIR}`
- `logging.run_name=${WANDB_RUN_NAME}`
- `logging.report_to_wandb=${REPORT_TO_WANDB}`
- `logging.entity=${WANDB_ENTITY}` when `WANDB_ENTITY` is set

Model checkpoint env vars remain recipe-specific, for example
`PRETRAINED_MODEL`, `SD3_PATH`, `QWEN_IMAGE_PATH`, and
`HUNYUAN_VIDEO15_PATH`.

## What Goes Where

| Layer | What belongs there |
|---|---|
| `conf/experiment/<exp>.yaml` | Model, algorithm, reward, rollout engine, sync, placement, batch geometry, LoRA/FSDP/EMA/NFT policy choices |
| YAML env interpolation | Checkpoint/data/output paths and logging identity when those are deployment-specific |
| `scripts/*.sh` | Python env activation, Ray startup, per-job path defaults, and forwarding CLI overrides |

Override precedence is:

```text
CLI Hydra override > launcher env var > YAML default
```

For different cluster geometry, override both placement and training batch
geometry together:

```bash
bash scripts/run_experiment_multinode.sh flowgrpo_fast_qwen_image_4x8 \
    placement.num_train_nodes=2 \
    placement.num_rollout_nodes=2 \
    training.topology.actor_count=16 \
    training.plan.local_batch_size=36 \
    training.plan.local_mini_batch_size=18
```

`validate_training_batch_geometry` reports inconsistent geometry before Ray
work starts.

## Compose Check

Use a real training recipe for config validation:

```bash
python -m diffusionrl.train +experiment=<experiment> --cfg job --resolve
```
