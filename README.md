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

- **Training Actors (Ray + TrainBackend)**: Responsible for the main training process through pluggable training backends (FSDP2 / VeOmni native; Megatron scaffold), reads `TrainingBatch` from the rollout buffer, and synchronizes parameters to the inference actors after training.
- **Inference Actors (FSDP / SGLang)**: Generates denoising trajectories and latent samples using pluggable sampling engines, producing lightweight `RolloutSamples`.
- **Reward Runtime**: Evaluates generated samples using configurable reward models (HPS, CLIP, PickScore, OCR, EditReward, etc.) through one actor-side precompute path with either local or HTTP-backed scoring.
- **RolloutServices + Rollout Functions**: The driver owns the outer rollout loop and directly holds `RolloutServices` plus the configured rollout/eval/reward hook callables. `RolloutServices` exposes request-level operations such as `build_request`, `plan_request_batches`, `execute_sampling_request`, `launch_sampling_request`, and `compute_advantages`; the configured rollout function (`rollout_function_dotpath`) is a first-class extension point for `prompt batch -> RolloutRequest -> sample -> reward hook`, and the driver entrypoint then computes advantages and assembles the final `TrainingBatch`.
- **Reward Hook**: Reward is a first-class rollout hook (`reward_hook_dotpath`) rather than a hard-coded workflow stage. The default hook reads scalar rewards that were already attached on the active sampling actor, while custom hooks can post-process or replace that behavior.

```
 ┌────────────────────┐     ┌────────────────────────────────────────┐
 │   Prompt Source    │────>│ Driver Entrypoint + RolloutServices    │
 └────────────────────┘     │ rollout_function_dotpath / reward_hook_dotpath │
                            │ plan RolloutRequest(s)                 │
                            └──────────────┬─────────────────────────┘
                                           │ actor_group.generate(request)
                                           v
                              ┌──────────────────────────┐
                              │   Inference ActorGroup   │
                              └─────────────┬────────────┘
                                            │ RolloutSamples
                                            v
                              ┌──────────────────────────┐
                              │ reward hook              │
                              └─────────────┬────────────┘
                                            │ rewards
                                            v
                              ┌──────────────────────────┐
                              │ driver: advantage +      │
                              │ assemble TrainingBatch   │
                              └─────────────┬────────────┘
                                            │ TrainingBatch
                                            v
                              ┌──────────────────────────┐
                              │      Rollout Buffer      │
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
| **Separate** | Inference and training on different GPU pools | Maximum throughput with sufficient GPUs |
| **Colocate** | Shared GPUs with explicit offload/onload | Memory-constrained environments |
| **Async Pipeline** | Rollout N+1 overlaps with training on N | Reduced idle time in separate mode |

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
For real datasets, symlink into `data/datasets/` and override `DATA_PATH`:

```bash
DATA_PATH=data/datasets/hpdv2/train.json \
  bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh --rollout.num-rollout 1
```

The user-facing dataset contract is prompt-only:

- preferred format: JSON / JSONL / TXT containing prompt strings
- JSON objects may use either `prompt` or `caption`
- extra per-sample fields are preserved as metadata for reward/eval
- precomputed embedding files are not a supported input path; if legacy manifests still contain embedding fields, they are ignored

For editing rewards such as `editreward`, keep the source image in sample metadata so
the reward scorer can compare `(source image, edited image, prompt)` triplets:

```json
{"prompt": "Add a green bowl on the branch", "metadata": {"source_image_path": "/path/to/source.png"}}
```

`editreward` accepts source-image metadata under common keys such as
`source_image_path`, `source_image`, `image_src`, `input_image_path`, and related aliases.

Default local paths are resolved against the repository root:

- `data_path`: `data/samples/prompts_toy.json`
- `eval_data_path`: unset by default; periodic eval then uses `data_path` with deterministic ordering
- `output_dir`: `outputs/`
- `sync.dir`: `outputs/weight_sync/`
- local model mount (optional symlink): `models/local/`

For a real validation split, pass `eval_data_path` explicitly. This keeps eval prompts independent from the training iterator and avoids train/eval data-stream coupling.

For external data / model directories, pass absolute paths directly (or create symlinks manually):

```bash
DATA_PATH=/path/to/external/data/train.json \
PRETRAINED_MODEL=/path/to/external/shared_models/sd3 \
bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh --rollout.num-rollout 1
```

### Training

We provide pre-configured training scripts for various algorithm + model combinations:

```bash
# FlowGRPO with SD3 (training-actor direct sampling)
bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh

# MixGRPO with SD3 (SGLang separate mode, rollout/training split)
bash scripts/train_mixgrpo_sd3_sglang_separate.sh

# NFT with SD3 (SGLang separate mode)
bash scripts/train_nft_sd3_sglang_separate.sh

# Override default parameters via environment variables
PRETRAINED_MODEL=/path/to/shared_models/sd3 \
DATA_PATH=/path/to/prompts.json \
ROLLOUT_GPUS=2 TRAINING_GPUS=2 BATCH_SIZE=2 \
    bash scripts/train_mixgrpo_sd3_sglang_separate.sh
```

Or use the CLI directly:

```bash
python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path black-forest-labs/FLUX.1-dev \
    --model.model-type flux \
    --algorithm.algorithm-type grpo \
    --algorithm.prompts-per-rollout 8 \
    --reward.reward-components hpsv2 \
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
    --rollout.output-dir outputs/my_experiment \
    --sync.protocol disabled
```

For asynchronous separate rollout/training overlap, use a dedicated async entry such as:

```bash
bash scripts/train_mixgrpo_sd3_sglang_separate.sh
```

For researcher work, start from `scripts/*.sh`.
Those shell templates are the primary maintained entry surface in this repo.
If you prefer grouped YAML editing, start from `scripts/example_flux_dancegrpo_sglang_separate.yaml`.
The local `configs/recipes/` directory may exist in some working trees, but it is
gitignored and is not the public repo interface.
The committed `scripts/example_flux_dancegrpo_sglang_separate.yaml` file is the
smallest public sanity-check config.
Public config tests cover that committed YAML in `scripts/`.

Optional YAML-driven entry examples:

```bash
python -m diffusionrl.train_async --config scripts/example_flux_dancegrpo_sglang_separate.yaml
```

Terminology:

- `rollout_id` / `rollout.step` is the outer rollout-train loop step.
- In practice it behaves similarly to a framework-level global step.
- It is not strictly the same as optimizer update count when gradient accumulation or inner epochs are enabled.

`--config` supports grouped YAML mappings (for example `algorithm: { ... }`, `training: { ... }`).
For rollout, use a grouped `rollout` YAML mapping (`rollout.mode`, `rollout.transport_dtype`,
`rollout.control`, `rollout.logging`, etc.) so the file shape matches the config sections in code.
Grouped YAML is now the only supported style for grouped fields.
Grouped CLI options also use dotted names (for example `--training.train-backend`).
Precision is grouped by runtime owner: use `precision.training.*` for training
model/FSDP/loss-forward precision, and `precision.rollout.*` for sampler/replay
autocast plus trajectory/logprob storage precision.
For dedicated SGLang rollout, the rollout-side prompt encoder also follows
`precision.rollout.autocast_precision`; do not configure a separate
prompt-encoder dtype under `rollout`.
`rollout.transport_dtype` remains a rollout transport setting,
not a `precision.*` field.
CLI options always override YAML. Unknown YAML keys fail fast by default; use
`--allow-unknown-config-keys` only when you intentionally want to ignore unknown keys.
`sync.protocol` must now be set explicitly: use `disabled` for `direct_sampling`,
and a dedicated-rollout sync mode (`tensor_payload`, `nccl_broadcast`, or
`checkpoint_path`) when rollout runs outside training actors.
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

`diffusionRL` now exposes three train backend entries:

- `fsdp`: default backend, implemented as **FSDP2** (`fully_shard` path).
- `veomni`: VeOmni native backend with FSDP2-focused data parallel mode (`data_parallel_mode=fsdp2`).
- `megatron`: interface scaffold for future integration (launcher/topology hooks are present, runtime path is not complete yet; requires a Megatron-specific actor class via `train_backend_kwargs.actor_class_path`).

Example: default FSDP2 training

```bash
python -m diffusionrl.train \
  --training.train-backend fsdp
```

Example: VeOmni-compatible FSDP2 mode

```bash
python -m diffusionrl.train \
  --training.train-backend veomni \
  --training.train-backend-kwargs '{"data_parallel_mode":"fsdp2"}'
```

Notes:

- FSDP2 runtime requires a torch build that provides composable FSDP2 APIs (`fully_shard`) and distributed checkpoint state-dict helpers.
- diffusionRL intentionally keeps VeOmni backend on `fsdp2` only for RL training.
- Built-in `veomni` backend calls VeOmni native APIs for model parallelization / optimizer / scheduler / grad clipping.

### Available Training Scripts

| Script | Algorithm | Model | Mode |
|--------|-----------|-------|------|
| `train_mixgrpo_sd3_sglang_separate.sh` | MixGRPO | SD3 | Separate (dedicated rollout actors, SGLang engine) |
| `train_nft_sd3_sglang_separate.sh` | NFT | SD3 | Separate (dedicated rollout actors, SGLang engine) |
| `train_flowgrpo_sd3_train_actor_sampling.sh` | FlowGRPO | SD3 | Training-actor direct sampling (FSDP engine) |

See [scripts/README.md](scripts/README.md) for exact per-script defaults.

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
4.  **Reward arguments**: `--reward.reward-components` for built-in scorers, `--reward.reward-dotpath` for custom scorer dotpaths, `--reward.reward-model-ckpt-path` for scorer checkpoints, `--reward.reward-backend {local,http}`, `--reward.local-reward-device`, and `--reward.reward-service-urls` for HTTP-backed scoring, etc. For the built-in `editreward` scorer, point `--reward.reward-model-ckpt-path` at the EditReward checkpoint directory, make sure each prompt sample carries source-image metadata as described above, and install `EditReward` (or set `EDITREWARD_PYTHON_PATH` to a local checkout).
5.  **Training arguments**: `--training.learning-rate`, `--training.num-updates-per-batch`, `--training.micro-batch-size`, `--training.max-grad-norm`, etc.
6.  **Runtime arguments**: `--ray.rollout-num-gpus-per-node`, `--ray.training-num-gpus-per-node`, `--ray.colocate-training-gpu-fraction`, `--ray.colocate-rollout-gpu-fraction`, `--ray.placement-strategy`, `--ray.offload-train`, `--ray.offload-rollout`, etc.

For config mechanics and current conventions, read
[diffusionrl/config/arguments.py](diffusionrl/config/arguments.py)
alongside `diffusionrl/config/argument_parsing.py`, `diffusionrl/config/validation.py`,
and `diffusionrl/config/resolution.py`. For runnable config examples, start with
[scripts/README.md](scripts/README.md) and the committed YAMLs under `scripts/`.

## Developer Guide

### Ray Layering

The Ray control plane is split into worker implementations and worker-group orchestration:

- `diffusionrl/ray/{rollout_actor.py,training_actor.py}`: single worker implementations
- `diffusionrl/ray/{rollout_group.py,training_group.py,group_factory.py}`: group orchestration and factories
- `diffusionrl/ray/ray_utils.py`: Ray distributed utilities + training-actor helper/service modules
- `diffusionrl/rollout/**`: driver-side rollout runtime, request planning/execution primitives, default rollout hooks
- `diffusionrl/training/**`: training execution + train backend logic
- `diffusionrl/distributed/**`: distributed coordination such as weight sync

The layer summary above is the current authoritative control-plane map.

### Project Structure

```
diffusionrl/
├── train.py / train_async.py      # Training entry points
├── types/                          # Canonical shared data types (RolloutRequest, RolloutSamples, TrainingBatch, Reward, Engine, SDE)
├── config/                         # Configuration system (TrainingArguments)
├── algorithms/                     # RL algorithms + advantage normalization helpers
├── samplers/                       # Inference engines (FSDP, SGLang)
│   ├── fsdp/                       #   FSDP-based: FluxSampler, SD3Sampler, HunyuanSampler
│   └── sglang/                     #   SGLang external service engine
├── reward/                         # Reward executors (local, HTTP, actor-side precompute)
├── models/                         # Model implementations (FLUX, SD3, Hunyuan, Mochi)
├── data/                           # Data loading and datasets
├── rollout/                        # Driver rollout runtime, request planning/execution, default hooks
├── training/                       # Training workflow, executor, update schedule, train backends
├── buffer/                         # Buffer subsystem (queue/filter/store)
├── distributed/                    # Distributed coordination (for example weight sync)
├── ray/                            # Ray distributed orchestration
│   ├── rollout_actor.py / training_actor.py
│   ├── rollout_group.py / training_group.py / group_factory.py
│   ├── buffer_actor.py / placement_group.py / ray_utils.py
├── patches/                        # Runtime patches (for example replay log-prob support)
└── utils/                          # Checkpointing, logging, EMA, media helpers
```

### Adding a Custom Algorithm

Subclass `BaseAlgorithm` and implement the current algorithm-centric contract:
`from_config()`, `get_sampling_requirements()`, reward/advantage handling, and
the training objective owned by the algorithm (`_loss_cls` / `compute_loss_and_backward()`).
For new code, always import shared data types from `diffusionrl.types` (single entry).

```python
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import BaseAlgorithm, SamplingRequirements
from diffusionrl.types import PromptEmbeddings, TimestepData


class _MyLoss:
    def __init__(self, algorithm: "MyAlgorithm") -> None:
        self.algorithm = algorithm

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del model, advantages, embeddings, kwargs
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder": True}


class MyAlgorithm(BaseAlgorithm):
    _loss_cls = _MyLoss

    @classmethod
    def from_config(cls, config: dict) -> "MyAlgorithm":
        extra = dict(config.get("algorithm_kwargs") or {})
        return cls(sde_ratio=float(extra.get("sde_ratio", 1.0)))

    def __init__(self, *, sde_ratio: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sde_ratio = float(sde_ratio)

    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
            extras={"sde_ratio": 1.0},
        )
```

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

- Built-in scorer: set `--reward.reward-components <builtin>` for one scorer, or
  `--reward.reward-components a,b --reward.reward-weights wa,wb` for multi-component reward.
- Custom scorer: set `--reward.reward-dotpath your_module.MyRewardScorer` to a full
  scorer class dotpath.
- Execution path: rewards are always computed on the active sampling actor.
  Use `--reward.reward-backend local` for in-process scorers, or
  `--reward.reward-backend http --reward.reward-service-urls ...` to call a
  remote reward service from the actor and return only scalar rewards to the driver.
  Local scoring defaults to `local_reward_device=cpu`; switch to `cuda` only when
  you intentionally want in-process GPU scoring on the sampling host.

Subclass `BaseRewardScorer` for the minimal contract:

```python
from diffusionrl.types.reward import RewardRequest, RewardResponse
from diffusionrl.reward.base import BaseRewardScorer

class MyRewardScorer(BaseRewardScorer):
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        # Your reward logic here
        ...
```

Then pass it via `--reward.reward-dotpath your_module.MyRewardScorer`.

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

Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
pip install ruff
ruff check diffusionrl/
ruff format diffusionrl/
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
