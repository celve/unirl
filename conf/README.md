# Configuration

DiffusionRL uses Hydra. The base config is `conf/train.yaml`; every training
run selects one recipe with:

```bash
python -m diffusionrl.train +experiment=<name>
```

`conf/experiment/<name>.yaml` is the source of truth for a recipe. Shell
launchers may provide path and logging defaults, but recipe-defining choices
belong in YAML.

## Composition Model

`conf/train.yaml` composes these default groups:

| Config section | Owns |
|---|---|
| `run` | outer-loop seed, data path, data source, rollout count, eval cadence |
| `model` | model bundle, checkpoint path, precision, LoRA target materialization |
| `algorithm` | driver-side rollout control such as samples per prompt and SDE index scheduling |
| `algorithms.<slot>` | train-side stage algorithms keyed by rollout trace slot |
| `sampling` | SDE strategy, step count, guidance, output shape, precision for trajectories/log-probs |
| `reward` | reward components, aggregation, local/remote reward runtime knobs |
| `rollout.engine` | trainside, SGLang, or vLLM-Omni rollout implementation |
| `rollout.plan` | request planning and forward batch sizing |
| `placement` | Ray placement and train/rollout GPU layout |
| `sync` | trainer-to-rollout weight sync; present only for dedicated rollout engines |
| `training.plan` | global/local batch sizes, mini-batch splitting, micro-batch slicing |
| `training.topology` | train actor count, DP size, replica/shard topology |
| `training.execution` | offload/onload, gradient accumulation, grad clipping |
| `training.policies` | ordered policy stack such as LoRA, FSDP, EMA, NFT LoRA |
| `logging` | wandb and report destinations |
| `resume` | output/checkpoint directory behavior |
| `evaluation` | evaluation cadence and eval data path |
| `debug` | debug mode selector; argparse debug runner is retired |

## Where Knobs Belong

Use this rule when adding or moving config:

| Change type | Put it in |
|---|---|
| New training recipe, model choice, algorithm choice, reward mix, placement, or batch geometry | `conf/experiment/<recipe>.yaml` |
| Cluster-local paths, model mounts, output directory, wandb identity | launcher env vars or CLI overrides |
| New typed runtime component | dataclass near the implementation, registered with `@register_config` |
| New reusable preset | `register_preset(...)` near the owning implementation |
| One-off debugging override | CLI dotted-key override |

Override precedence is:

```text
CLI Hydra override > launcher env var > YAML default
```

Examples:

```bash
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate \
    run.num_rollouts=20 \
    training.plan.local_batch_size=24

DATA_PATH=/abs/path/train.json \
OUTPUT_DIR=/abs/path/outputs/run1 \
bash scripts/run_experiment_single_node.sh grpo_wan21_t2v
```

## Runtime Contracts

Cross-component validators run before Ray actor creation:

- direct sampling (`rollout/engine: trainside`) forbids `sync:`;
- dedicated rollout engines (`sglang`, `vllm_omni`) require `sync:`;
- CUDA-IPC sync is only valid with vLLM-Omni rollout;
- direct sampling forbids train/rollout offload flags;
- LoRA target modules are materialized from the selected model bundle when omitted;
- training batch geometry must divide cleanly across the declared topology.

Use a compose check before launching a large job:

```bash
python -m diffusionrl.train +experiment=<name> --cfg job --resolve
```

## Recipe Inventory

Current maintained experiment recipes:

| Family | Experiments |
|---|---|
| SD3 FlowGRPO | `flowgrpo_fast_sd3`, `flowgrpo_fast_sd3_trainside`, `flowgrpo_fast_sd3_colocate`, `flowgrpo_fast_sd3_colocate_reproduce`, `flowgrpo_fast_sd3_colocate_reproduce_4x8` |
| SD3 SGLang GRPO | `flowgrpo_sd3_sglang_native_colocate`, `flowgrpo_sd3_sglang_native_separate`, `flowgrpo_sd3_sglang_replay_colocate`, `flowgrpo_sd3_sglang_replay_separate` |
| SD3 NFT | `nft_sd3`, `nft_sd3_reward_service`, `nft_sd3_sglang` |
| SD3 Dance/Mix | `dancegrpo_fast_sd3`, `mixgrpo_sd35` |
| Qwen-Image FlowGRPO | `flowgrpo_fast_qwen_image`, `flowgrpo_fast_qwen_image_2x8`, `flowgrpo_fast_qwen_image_4x8` |
| Qwen-Image NFT | `nft_qwen_image`, `nft_qwen_image_2x8`, `nft_qwen_image_4x8` |
| WAN T2V | `grpo_wan21_t2v`, `grpo_wan22_t2v_14b` |
| WAN I2V | `grpo_wan21_i2v`, `grpo_wan22_i2v` |
| HunyuanVideo | `grpo_hunyuan_video15_t2v_videopickscore` |
| HunyuanImage3 | `hi3_think_recaption_colocate` |
