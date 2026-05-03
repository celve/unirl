# DiffusionRL

**DiffusionRL** is a distributed reinforcement learning framework for diffusion model optimization, providing two core capabilities:

1.  **High-Performance Distributed Training**: Supports efficient RL training in various deployment modes (separate, colocate, async pipeline) by orchestrating Ray-based inference and training actor pools;
2.  **Flexible Algorithm & Engine Integration**: Enables pluggable RL algorithms (GRPO, MixGRPO, DanceGRPO, FlowGRPO, NFT), sampling engines (FSDP, SGLang), and reward models through a unified contract-driven architecture.

DiffusionRL supports the following diffusion models:
- **Image**: FLUX.1 (Black Forest Labs), Stable Diffusion 3 (Stability AI);
- **Video**: Hunyuan Video (Tencent), Mochi (Genmo).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Supported Algorithms](#supported-algorithms)
- [Arguments Walkthrough](#arguments-walkthrough)
- [Developer Guide](#developer-guide)
- [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

**Module Descriptions**:

- **Training Actors (Ray + TrainBackend)**: Responsible for the main training process through pluggable training backends (FSDP2; Megatron scaffold), reads `TrainingBatch` from the rollout buffer, and synchronizes parameters to the inference actors after training.
- **Inference Actors (FSDP / SGLang)**: Generates denoising trajectories and latent samples using pluggable sampling engines, producing lightweight `RolloutSamples`.
- **Reward Runtime**: Evaluates generated samples using configurable reward models (HPS, CLIP, PickScore, OCR, etc.) through one actor-side precompute path with local in-process scoring.
- **RolloutPipeline**: The driver owns the outer rollout loop via `RolloutPipeline`, which drives the `load_prompts -> plan_requests -> exec_request -> aggregate -> convert_training_data` phases. Reward scoring and advantage computation run inside the rollout actor through `RolloutActor.run_rollout_pipeline`, and the driver then assembles the final `TrainingBatch`.

```
 ┌────────────────────┐     ┌────────────────────────────────────────┐
 │   Prompt Source    │────>│ Driver Entrypoint + RolloutPipeline    │
 └────────────────────┘     │ plan RolloutRequest(s)                 │
                            └──────────────┬─────────────────────────┘
                                           │ actor_group.generate(request)
                                           v
                              ┌──────────────────────────┐
                              │   Inference ActorGroup   │
                              │ (sample + reward +       │
                              │  advantage)              │
                              └─────────────┬────────────┘
                                            │ TrainingBatch
                                            v
                              ┌──────────────────────────┐
                              │    Training ActorGroup   │
                              └─────────────┬────────────┘
                                            │ weight sync
                                            v
                                   Back to Inference
```

**Deployment Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Direct Sampling** | Training actors also perform sampling | Simplest topology when one backend can do both |
| **Separate** | Inference and training on different GPU pools | Maximum throughput with sufficient GPUs |
| **Colocate** | Shared GPUs with explicit offload/onload | Memory-constrained environments |

## Quick Start

### Installation

We support two equivalent core installation paths.
Python `>=3.10` is required (the SGLang diffusion fork depends on it).

```bash
# Option A (recommended): editable install with core extras
pip install -e ".[train,infer,eval]" --no-build-isolation

# Option B: requirements-based install (same dependency set as Option A)
pip install -r requirements.txt --no-build-isolation
pip install -e . --no-deps --no-build-isolation
```

To match our validated GPU environment, install FlashAttention explicitly:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For development tools:

```bash
pip install -e ".[train,infer,eval,dev]"
```

If your environment has complete build tooling/mirrors, you can omit `--no-build-isolation`:

```bash
pip install -e ".[train,infer,eval]"
# or
pip install -r requirements.txt
pip install -e . --no-deps
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`mmcv` / `mmdetection` are intentionally **not** installed by default.
Install them only when needed (Geneval/OpenMMLab workflows):

- [docs/geneval_mmcv_setup.md](docs/geneval_mmcv_setup.md)

### Data Preparation

Toy data is included in `data/samples/` for smoke tests.
For real datasets, symlink into `data/datasets/` and override `DATA_PATH` when invoking a `reproduce_scripts/` wrapper.

The user-facing dataset contract is prompt-only:

- preferred format: JSON / JSONL / TXT containing prompt strings
- JSON objects may use either `prompt` or `caption`
- extra per-sample fields are preserved as metadata for reward/eval
- precomputed embedding files are not a supported input path; if legacy manifests still contain embedding fields, they are ignored

Default local paths are resolved against the repository root:

- `data_path`: `data/samples/prompts_toy.json`
- `eval_data_path`: unset by default; periodic eval then uses `data_path` with deterministic ordering
- `output_dir`: `outputs/`
- `sync.dir`: `outputs/weight_sync/`
- local model mount (optional symlink): `models/local/`

For a real validation split, pass `eval_data_path` explicitly. This keeps eval prompts independent from the training iterator and avoids train/eval data-stream coupling.

For external data / model directories, pass absolute paths directly (or create symlinks manually) via `DATA_PATH` / `PRETRAINED_MODEL` when invoking a `reproduce_scripts/` wrapper.

### Training

See `reproduce_scripts/` for per-recipe wrappers; all delegate to `scripts/train.sh` (or `scripts/train_tq.sh` for TransferQueue variants).

Or use the CLI directly:

```bash
python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path black-forest-labs/FLUX.1-dev \
    --model.model-type flux \
    --algorithm.algorithm-type grpo \
    --algorithm.prompts-per-rollout 8 \
    --reward.definition.components "[{model_name: hpsv2, weight: 1.0}]" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path data/samples/prompts_toy.json \
    --eval-data-path data/samples/prompts_toy.json \
    --sampling.sde-type dance \
    --sampling.eta 0.3 \
    --sampling.num-inference-steps 25 \
    --training.use-lora true \
    --training.lora-rank 16 \
    --precision.training.model-precision bf16 \
    --precision.training.autocast-precision bf16 \
    --precision.rollout.autocast-precision bf16 \
    --precision.rollout.trajectory-precision fp16 \
    --precision.rollout.logprob-precision fp32 \
    --rollout.mode direct_sampling \
    --rollout.num-rollout 300 \
    --rollout.output-dir outputs/my_experiment
```

For researcher work, start from `reproduce_scripts/*.sh` — those per-recipe wrappers are the primary maintained entry surface and delegate to the generic launchers `scripts/train.sh` / `scripts/train_tq.sh`.
The local `configs/recipes/` directory may exist in some working trees, but it is
gitignored and is not the public repo interface.

Terminology:

- `rollout_id` / `rollout.step` is the outer rollout-train loop step.
- In practice it behaves similarly to a framework-level global step.
- It is not strictly the same as optimizer update count when gradient accumulation or inner epochs are enabled.

`--config` supports grouped YAML mappings (for example `algorithm: { ... }`, `training: { ... }`).
For rollout, use a grouped `rollout` YAML mapping (`rollout.mode`,
`rollout.control`, `rollout.logging`, etc.) so the file shape matches the config sections in code.
Grouped YAML is now the only supported style for grouped fields.
Grouped CLI options also use dotted names (for example `--training.train-backend`).
Precision is grouped by runtime owner: use `precision.training.*` for training
model/FSDP/loss-forward precision, and `precision.rollout.*` for sampler/replay
autocast plus trajectory/logprob storage precision.
For dedicated SGLang rollout, the rollout-side prompt encoder also follows
`precision.rollout.autocast_precision`; do not configure a separate
prompt-encoder dtype under `rollout`.
CLI options always override YAML. Unknown YAML keys fail fast by default; use
`--allow-unknown-config-keys` only when you intentionally want to ignore unknown keys.
Weight sync is opt-in: omit `cfg.sync` for `direct_sampling` (no rollout actors to push to),
and select a dedicated-rollout sync variant (`+sync=tensor_payload`, `+sync=nccl_broadcast`,
or `+sync=checkpoint_path`) when rollout runs outside training actors. The cross-component
validator enforces this biconditional (`engine=fsdp` ⇔ no `sync`; `engine=sglang` ⇔ `sync` set).
In `direct_sampling`, leave `rollout.rollout_engine` unset.
Direct sampling is selected only by `rollout.mode=direct_sampling`;
dedicated rollout-only fields must remain unset there.
Config docs are descriptive rather than normative: current behavior lives under
`diffusionrl/config/*`, and docs/examples should be updated alongside config refactors.

Training geometry is rollout-driven only:

- `algorithm.prompts_per_rollout * algorithm.samples_per_prompt` defines the global rollout batch.
- The resolved local training batch is derived from the training topology.
- `training.num_updates_per_batch` controls how one resolved local batch is split across optimizer updates.
- `training.local_mini_batch_size` is a derived value, computed as `local_batch_size / num_updates_per_batch`.
- `training.micro_batch_size` only controls micro-step slicing inside one local mini-batch and must evenly divide the resolved `local_mini_batch_size`.

### Training Backend Selection

`diffusionRL` now exposes two train backend entries:

- `fsdp`: default backend, implemented as **FSDP2** (`fully_shard` path).
- `megatron`: interface scaffold for future integration (launcher/topology hooks are present, runtime path is not complete yet; requires a Megatron-specific actor class via `train_backend_kwargs.actor_class_path`).

Example: default FSDP2 training

```bash
python -m diffusionrl.train \
  --training.train-backend fsdp
```

Notes:

- FSDP2 runtime requires a torch build that provides composable FSDP2 APIs (`fully_shard`) and distributed checkpoint state-dict helpers.

## Supported Algorithms

| Algorithm | Description | Transition Rule | Key Feature |
|-----------|-------------|----------|-------------|
| **GRPO** | Group Relative Policy Optimization | `flow`, `cps`, `dpm2` | Standard trajectory-based RL |
| **DanceGRPO** | GRPO with dance-specific SDE formulation | `dance` | Optimized for FLUX/SD3 |
| **FlowGRPO** | Flow matching formulation | `flow` | Flow-based objective |
| **MixGRPO** | Mixed ODE/SDE sampling | Configurable | Flexible SDE ratio (0~1) with window scheduler |
| **NFT** | Negative Fine-Tuning | N/A | Forward diffusion, no trajectory needed |

## Arguments Walkthrough

Arguments in DiffusionRL are organized into the following categories:

1.  **Model arguments**: `--model.model-type`, `--model.pretrained-model-ckpt-path`, `--training.use-lora`, `--training.lora-rank`, `--training.lora-alpha`, etc.
2.  **Sampling arguments**: `--sampling.sde-type`, `--sampling.eta`, `--sampling.num-inference-steps`, `--sampling.guidance-scale`, `--sampling.shift`, `--sampling.timestep-fraction`, etc.
3.  **Algorithm arguments**: `--algorithm.algorithm-dotpath`, `--algorithm.prompts-per-rollout`, `--algorithm.samples-per-prompt`, plus shared typed fields such as `--algorithm.adv-normalization`, `--algorithm.eval-ema-decay`, and `--algorithm.shuffle-samples`. Use repeated `--algorithm.kwarg key=value` only for true algorithm-specific extension keys. In YAML, put those extension keys under `algorithm.algorithm_kwargs`.
4.  **Reward arguments**: `--reward.definition.components` for built-in scorers (typed `{model_name, weight}` records), `--reward.provider.dotpath` for custom scorer dotpaths, `--reward.provider.model-ckpt-path` for scorer checkpoints, `--reward.execution.local-device`, etc.
5.  **Training arguments**: `--training.learning-rate`, `--training.num-updates-per-batch`, `--training.micro-batch-size`, `--training.max-grad-norm`, etc.
6.  **Runtime arguments**: `--ray.rollout-num-gpus-per-node`, `--ray.training-num-gpus-per-node`, `--ray.colocate-training-gpu-fraction`, `--ray.colocate-rollout-gpu-fraction`, `--ray.placement-strategy`, `--ray.offload-train`, `--ray.offload-rollout`, etc.

For config mechanics and current conventions, read
[diffusionrl/config/arguments.py](diffusionrl/config/arguments.py)
alongside `diffusionrl/config/argument_parsing.py`, `diffusionrl/config/validation.py`,
and `diffusionrl/config/resolution.py`. For runnable config examples, start with
[scripts/README.md](scripts/README.md) and the experiment YAMLs under `conf/experiment/`.

## Developer Guide

### Ray Layering

The Ray control plane is split into worker implementations and worker-group orchestration:

- `diffusionrl/ray/{rollout_actor.py,train_actor.py}`: single worker implementations
- `diffusionrl/ray/group/{rollout.py,train.py}`: group orchestration (spawn + dispatch + control plane)
- `diffusionrl/ray/utils/`: stateless helpers imported by actors (`net`, `gpu`, `node`)
- `diffusionrl/rollout/**`: driver-side `RolloutPipeline` and request-builder helpers
- `diffusionrl/training/**`: training execution + train backend logic
- `diffusionrl/distributed/**`: distributed coordination such as weight sync

The layer summary above is the current authoritative control-plane map.

### Project Structure

```
diffusionrl/
├── train.py                        # Training entry point
├── types/                          # Canonical shared data types (RolloutRequest, RolloutSamples, TrainingBatch, Reward, Engine, SDE)
├── config/                         # Configuration system (TrainingArguments)
├── algorithms/                     # RL algorithms + advantage normalization helpers
├── samplers/                       # Inference engines (FSDP, SGLang)
│   ├── fsdp/                       #   FSDP-based: FluxSampler, SD3Sampler, HunyuanSampler
│   └── sglang/                     #   SGLang external service engine
├── reward/                         # Reward executors (local in-process, actor-side precompute)
├── models/                         # Model implementations (FLUX, SD3, Hunyuan, Mochi)
├── data/                           # Data loading and datasets
├── rollout/                        # Driver rollout runtime, request planning/execution, default hooks
├── training/                       # Training workflow, executor, update schedule, train backends
├── buffer/                         # Buffer subsystem (queue/filter/store)
├── distributed/                    # Distributed coordination (for example weight sync)
├── ray/                            # Ray distributed orchestration
│   ├── rollout_actor.py / train_actor.py
│   ├── group/                      #   rollout.py / train.py (group orchestration)
│   ├── buffer_actor.py / placement_group.py
│   ├── utils/                      #   net/gpu/node stateless helpers
├── patches/                        # Runtime patches (for example replay log-prob support)
└── utils/                          # Checkpointing, logging, EMA, media helpers
```

### Adding a Custom Algorithm

Subclass `BaseAlgorithm` and implement the current algorithm-centric contract:
`__init__(*, config)`, `get_sampling_requirements()`, reward/advantage handling, and
the training objective owned by the algorithm (`compute_loss_and_backward()`).
For new code, always import shared data types from `diffusionrl.types` (single entry).
See `diffusionrl_plugins/algorithms/minimal_algorithm.py` for a complete example.

Then pass it via `--algorithm.algorithm-dotpath your_module.MyAlgorithm`.
See the fully working reference implementation:
`diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm`.
For the current extension contract, read [`diffusionrl/algorithms/base.py`](diffusionrl/algorithms/base.py)
and the minimal reference implementation in
`diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm`.

### Plugin Templates (diffusionrl_plugins)

This repo ships minimal templates under `diffusionrl_plugins/` for common extension
points:

- Model: `diffusionrl_plugins.models.wan21.Wan21ModelBundle`
- Sampler: `diffusionrl_plugins.samplers.minimal_sampler.MinimalSampler`
- Algorithm: `diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm`
- Reward: `diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer`
Notes:
- There is no plugin auto-registration; pass full dotpaths via CLI args.
- `--model.model-type <name>` short-name resolution works only when the model class
  declares `declared_model_type()`, `default_sampler_path()`, and
  `default_sampler_engine()`.

### Adding a Custom Reward Scorer

Use reward config in three layers:

- Built-in scorer: set `--reward.definition.components "[{model_name: <builtin>, weight: 1.0}]"`
  for one scorer, or pass multiple `{model_name, weight}` records for multi-component reward.
- Custom scorer: set `--reward.provider.dotpath your_module.MyRewardScorer` to a full
  scorer class dotpath.
- Execution path: rewards are always computed on the active sampling actor
  using in-process scorers. Local scoring defaults to
  `reward.execution.local_device=cpu`; switch to `cuda` only when you
  intentionally want in-process GPU scoring on the sampling host.

Subclass `BaseRewardScorer` for the minimal contract:

```python
from diffusionrl.types.reward import RewardRequest, RewardResponse
from diffusionrl.reward.base import BaseRewardScorer

class MyRewardScorer(BaseRewardScorer):
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        # Your reward logic here
        ...
```

Then pass it via `--reward.provider.dotpath your_module.MyRewardScorer`.

If you want the built-in local scorer lifecycle helpers (`device`, eager model
load, standard `offload()` / `onload()` behavior), subclass
`diffusionrl.reward.scorers.base_local.BaseLocalRewardScorer` instead.

### Running Tests

```bash
# CLI parse smoke check
python -m diffusionrl.train --help

# Python syntax check
python -m compileall -q diffusionrl

# Script syntax check
for f in scripts/*.sh; do bash -n "$f"; done
```

### Code Style

Install pre-commit hooks to match the CI lint gate (runs ruff + several safety checks on every commit):

```bash
pip install pre-commit
pre-commit install            # runs on every `git commit`
pre-commit run --all-files    # manually run against the whole repo
```

- Contributions are welcome! Feel free to submit an Issue or PR.

## FAQ & Acknowledgements

- Special thanks to the following projects & communities: [DanceGRPO](https://github.com/jwhj/DanceGRPO), [diffusers](https://github.com/huggingface/diffusers), [Ray](https://github.com/ray-project/ray), [FastVideo](https://github.com/hao-ai-lab/FastVideo), [SGLang](https://github.com/sgl-project/sglang), [PEFT](https://github.com/huggingface/peft), and others.
- To cite DiffusionRL, please use:

```bibtex
@misc{diffusionrl_github,
  title        = {DiffusionRL: A Distributed RL Framework for Diffusion Model Optimization},
  year         = {2025},
  howpublished = {\url{https://github.com/your-org/diffusionrl}},
  note         = {GitHub repository},
}
```
