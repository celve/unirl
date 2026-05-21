# Training Scripts

`scripts/` is intentionally thin. Training semantics live in
`conf/experiment/*.yaml`; shell scripts only prepare the runtime, start Ray,
set path/logging defaults, and forward Hydra overrides.

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

## Config Inventory

These are training recipes, not smoke/debug fixtures.

| Family | Experiments | Notes |
|---|---|---|
| SD3 FlowGRPO | `flowgrpo_fast_sd3`, `flowgrpo_fast_sd3_trainside`, `flowgrpo_fast_sd3_colocate`, `flowgrpo_fast_sd3_colocate_reproduce`, `flowgrpo_fast_sd3_colocate_reproduce_4x8` | SD3 GRPO variants across trainside, vLLM-Omni, colocate, and reproduce layouts |
| SD3 SGLang GRPO | `flowgrpo_sd3_sglang_native_colocate`, `flowgrpo_sd3_sglang_native_separate`, `flowgrpo_sd3_sglang_replay_colocate`, `flowgrpo_sd3_sglang_replay_separate` | Dedicated rollout engine variants |
| SD3 NFT | `nft_sd3`, `nft_sd3_reward_service`, `nft_sd3_sglang` | PR #96 lineage, mapped to current NFT policy stack |
| SD3 Dance/Mix | `dancegrpo_fast_sd3`, `mixgrpo_sd35` | PR #109 training recipes; Dance uses `sampling/sde_strategy=dance`, Mix uses windowed Flow SDE indices |
| Qwen-Image | `flowgrpo_fast_qwen_image`, `flowgrpo_fast_qwen_image_2x8`, `flowgrpo_fast_qwen_image_4x8` | PR #104 lineage; node-size geometry is YAML, not bash |
| Qwen-Image NFT | `nft_qwen_image`, `nft_qwen_image_2x8`, `nft_qwen_image_4x8` | Qwen-Image forward-process NFT recipes |
| WAN T2V | `grpo_wan21_t2v`, `grpo_wan22_t2v_14b` | PR #96 lineage, current WAN21/WAN22 model packages |
| WAN I2V | `grpo_wan21_i2v`, `grpo_wan22_i2v` | PR #103 lineage; image-conditioned data source path required |
| HunyuanVideo | `grpo_hunyuan_video15_t2v_videopickscore` | PR #101 lineage |
| HunyuanImage3 | `hi3_think_recaption_colocate` | HunyuanImage3 diffusion + AR recipe |

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
