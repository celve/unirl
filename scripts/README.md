# Training Scripts

`scripts/` is intentionally thin. Training semantics live in
`conf/experiment/*.yaml`; shell scripts only prepare the runtime, start Ray,
set path/logging defaults, and forward Hydra overrides.

For the experiment inventory and config ownership rules, see
`../conf/README.md`.

## Launchers

| Script | Use |
|---|---|
| `run_experiment_single_node.sh <experiment>` | Cluster-agnostic 1-node launcher for any `conf/experiment/<experiment>.yaml` (no platform env needed) |
| `run_experiment_multinode_taiji.sh <experiment>` | Multi-node launcher for the **taiji** platform; rank 0 runs the training driver, other ranks join Ray |

Examples:

```bash
bash scripts/run_experiment_single_node.sh grpo_wan21_t2v
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate
bash scripts/run_experiment_multinode_taiji.sh flowgrpo_fast_qwen_image_2x8
bash scripts/run_experiment_multinode_taiji.sh nft_qwen_image_4x8 run.num_rollouts=100
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

## Data plane

Both launchers pick how rollout samples reach the trainer with `DATA_PLANE`:

| `DATA_PLANE` | Effect | Extra infra |
|---|---|---|
| `ray` (default) | Driver gathers rollouts over the Ray object store | none |
| `tq_simple` | TransferQueue on Ray-backed host storage (off-driver) | none |
| `tq_mooncake` | TransferQueue over mooncake (RDMA across nodes) | external `mooncake_master` + `http_metadata_server`; set `MOONCAKE_METADATA_URL`, `MOONCAKE_MASTER_ADDR`, `PROTOCOL` (rdma/tcp) |
| `keep_local` | Direct-sampling actors keep rollouts local; only light metadata crosses to the driver | none |

```bash
DATA_PLANE=tq_simple bash scripts/run_experiment_single_node.sh <experiment>

DATA_PLANE=tq_mooncake PROTOCOL=rdma \
  MOONCAKE_METADATA_URL=http://$CHIEF_IP:8080/metadata \
  MOONCAKE_MASTER_ADDR=$CHIEF_IP:50051 \
  bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
```

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
bash scripts/run_experiment_multinode_taiji.sh flowgrpo_fast_qwen_image_4x8 \
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
