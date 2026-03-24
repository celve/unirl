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
- **Inference Actors (FSDP / SGLang)**: Generates denoising trajectories and latent samples using pluggable sampling engines, producing `RolloutOutput` with a unified v1 contract.
- **Reward Runtime**: Evaluates generated samples using configurable reward models (HPS, CLIP, PickScore, OCR, etc.) with a clean split between reward semantics and execution placement.
- **RolloutManager + RolloutWorkflow**: The rollout-side producer boundary. `RolloutManager` owns rollout-local services and public Ray-facing entrypoints; `RolloutWorkflow` owns the readable `sample -> reward -> advantage -> assemble` business chain before data is handed to the rollout buffer.

```
 ┌────────────────────┐     ┌─────────────────────────────┐
 │   Prompt Source     │────>│ RolloutManager / Producer   │
 └────────────────────┘     │   local services + facade    │
                            │        RolloutWorkflow       │
                            │ sample -> reward -> assemble │
                            └──────────────┬───────────────┘
                                           │ BufferPayload
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
  bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh --rollout.control.num-rollout 1
```

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

For external data / model directories, pass absolute paths directly (or create symlinks manually):

```bash
DATA_PATH=/path/to/external/data/train.json \
PRETRAINED_MODEL=/path/to/external/shared_models/flux \
bash scripts/train_dancegrpo_flux_train_actor_sampling.sh --rollout.control.num-rollout 1
```

### Training

We provide pre-configured training scripts for various algorithm + model combinations:

```bash
# DanceGRPO with FLUX (SGLang separate mode, rollout/training split)
bash scripts/train_dancegrpo_flux_sglang_separate.sh

# MixGRPO with SD3 (SGLang separate mode)
bash scripts/train_mixgrpo_sd3_sglang_separate.sh

# NFT with SD3 (training-actor sampling mode)
bash scripts/train_nft_sd3_train_actor_sampling.sh

# Override default parameters via environment variables
NUM_ROLLOUT=100 BATCH_SIZE=2 ROLLOUT_GPUS=4 TRAINING_GPUS=4 \
    bash scripts/train_dancegrpo_flux_sglang_separate.sh
```

Or use the CLI directly:

```bash
python -m diffusionrl.train \
    --model.pretrained-model-saved-path black-forest-labs/FLUX.1-dev \
    --model.model-type flux \
    --algorithm.algorithm-type grpo \
    --algorithm.prompts-per-rollout 8 \
    --reward.reward-model-name hpsv2 \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path data/samples/prompts_toy.json \
    --eval-data-path data/samples/prompts_toy.json \
    --sampling.sde-type dance \
    --sampling.eta 0.3 \
    --sampling.num-inference-steps 25 \
    --training.use-lora true \
    --training.lora-rank 16 \
    --rollout.topology.mode direct_rollout \
    --rollout.control.num-rollout 300 \
    --rollout.artifacts.output-dir outputs/my_experiment \
    --sync.protocol disabled
```

For asynchronous separate rollout/training overlap, use the dedicated async entry:

```bash
bash scripts/train_dancegrpo_flux_sglang_separate.sh
```

For researcher work, start from `scripts/*.sh`.
Those shell templates are the primary maintained entry surface in this repo.
If you prefer grouped YAML editing, use `scripts/example_*.yaml` as auxiliary examples.
The local `configs/recipes/` directory may exist in some working trees, but it is
gitignored and is not the public repo interface.
The `scripts/minimal_*.yaml` files remain the smallest sanity-check configs.
Public config tests cover the committed YAMLs in `scripts/`.

Optional YAML-driven entry examples:

```bash
python -m diffusionrl.train --config scripts/example_flux_dancegrpo_direct.yaml
python -m diffusionrl.train_async --config scripts/example_flux_dancegrpo_sglang_separate.yaml
python -m diffusionrl.train --config scripts/example_hunyuan_dancegrpo_direct.yaml --rollout.control.num-rollout 100
```

Terminology:

- `rollout_id` / `rollout.step` is the outer rollout-train loop step.
- In practice it behaves similarly to a framework-level global step.
- It is not strictly the same as optimizer update count when gradient accumulation or inner epochs are enabled.

`--config` supports grouped YAML mappings (for example `algorithm: { ... }`, `training: { ... }`).
For rollout, use nested sections such as `rollout.topology`, `rollout.control`,
and `rollout.logging` so the file shape matches the config sections in code.
Grouped YAML is now the only supported style for grouped fields.
Grouped CLI options also use dotted names (for example `--training.train-backend`).
CLI options always override YAML. Unknown YAML keys fail fast by default; use
`--allow-unknown-config-keys` only when you intentionally want to ignore unknown keys.
`sync.protocol` must now be set explicitly: use `disabled` for `direct_rollout`,
and a dedicated-rollout sync mode (`tensor_payload`, `nccl_broadcast`, or
`checkpoint_path`) when rollout runs outside training actors.
In `direct_rollout`, leave `rollout.topology.service_engine` unset.
Direct sampling is selected only by `rollout.topology.mode=direct_rollout`;
dedicated rollout-only fields must remain unset there.
When `rollout.buffer.reassemble_by_group=true`,
`rollout.buffer.group_size` must be set explicitly.
Config docs are descriptive rather than normative: current behavior lives under
`diffusionrl/config/*`, and docs/examples should be updated alongside config refactors.

Training geometry is rollout-driven only:

- `algorithm.prompts_per_rollout * algorithm.samples_per_prompt` defines the global rollout batch.
- The resolved local training batch is derived from the training topology.
- `training.num_updates_per_local_batch` controls how one resolved local batch is split across optimizer updates.
- `training.local_update_batch_size` is a derived value, computed as `local_batch_size / num_updates_per_local_batch`.
- `training.local_micro_batch_size` only controls micro-step slicing inside one local update batch and must evenly divide the resolved `local_update_batch_size`.

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
| `train_*_sglang_separate.sh` | DanceGRPO / MixGRPO / FlowGRPO / NFT | FLUX / SD3 / Hunyuan | Separate (dedicated rollout actors, SGLang engine) |
| `train_dancegrpo_flux_sglang_colocate.sh` | DanceGRPO | FLUX | Colocate (SGLang engine) |
| `train_*_train_actor_sampling.sh` | DanceGRPO / MixGRPO / FlowGRPO / NFT | FLUX / SD3 / Hunyuan | Training-actor direct sampling (FSDP engine) |
| `train_plugin_demo.sh` | Plugin demo | FLUX | Training-actor sampling |

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

1.  **Model arguments**: `--model.model-type`, `--model.pretrained-model-saved-path`, `--training.use-lora`, `--training.lora-rank`, `--training.lora-alpha`, etc.
2.  **Sampling arguments**: `--sampling.sde-type`, `--sampling.eta`, `--sampling.num-inference-steps`, `--sampling.guidance-scale`, `--sampling.shift`, `--sampling.timestep-fraction`, etc.
3.  **Algorithm arguments**: `--algorithm.algorithm-path`, `--algorithm.prompts-per-rollout`, `--algorithm.samples-per-prompt`, plus shared typed fields such as `--algorithm.adv-normalization`, `--algorithm.eval-ema-decay`, and `--algorithm.shuffle-samples`. Use repeated `--algorithm.kwarg key=value` only for true algorithm-specific extension keys. In YAML, put those extension keys under `algorithm.algorithm_kwargs`.
4.  **Reward arguments**: `--reward.reward-model-name` / `--reward.reward-models` for built-in scorers, `--reward.reward-path` for custom scorer dotpaths, `--reward.reward-location` for manager vs sampling-actor execution, `--reward.use-http-reward` + `--reward.reward-service-url` for HTTP-backed scoring, etc.
5.  **Training arguments**: `--training.learning-rate`, `--training.num-updates-per-local-batch`, `--training.local-micro-batch-size`, `--training.max-grad-norm`, etc.
6.  **Runtime arguments**: `--ray.rollout-num-gpus-per-node`, `--ray.training-num-gpus-per-node`, `--ray.colocate-training-gpu-fraction`, `--ray.colocate-rollout-gpu-fraction`, `--ray.placement-strategy`, `--ray.offload-train`, `--ray.offload-rollout`, etc.

For config mechanics and current conventions, start with
[docs/Parameter_System_Guide.md](docs/Parameter_System_Guide.md).
For the full argument reference, read [diffusionrl/config/arguments.py](diffusionrl/config/arguments.py)
alongside `diffusionrl/config/argument_parsing.py`, `diffusionrl/config/validation.py`,
and `diffusionrl/config/resolution.py`.

## Developer Guide

### Ray Layering

The Ray control plane is split into worker implementations and worker-group orchestration:

- `diffusionrl/ray/{rollout_actor.py,training_actor.py}`: single worker implementations
- `diffusionrl/ray/{rollout_group.py,training_group.py,group_factory.py}`: group orchestration and factories
- `diffusionrl/ray/ray_utils.py`: Ray distributed utilities + training-actor helper/service modules
- `diffusionrl/ray/rollout_manager.py`: control-plane actor
- `diffusionrl/orchestration/**`: rollout/eval/train workflow logic
- `diffusionrl/training/**`: training execution + train backend logic
- `diffusionrl/distributed/**`: distributed coordination such as weight sync

Detailed layer diagram:
- [docs/Ray_Layering.md](docs/Ray_Layering.md)

### Project Structure

```
diffusionrl/
├── train.py / train_async.py      # Training entry points
├── types/                          # Canonical shared data types (RolloutOutput, TrainingBatch, Reward, Engine, SDE)
├── config/                         # Configuration system (TrainingArguments)
├── algorithms/                     # RL algorithms + advantage normalization helpers
├── samplers/                       # Inference engines (FSDP, SGLang)
│   ├── fsdp/                       #   FSDP-based: FluxSampler, SD3Sampler, HunyuanSampler
│   └── sglang/                     #   SGLang external service engine
├── reward/                         # Reward executors (local, HTTP, Ray service, actor-local precompute)
├── models/                         # Model implementations (FLUX, SD3, Hunyuan, Mochi)
├── data/                           # Data loading and datasets
├── orchestration/                  # Rollout / eval / train business workflows
│   ├── request_builder.py          #   typed rollout request expansion/sub-batching
│   ├── rollout_workflow.py         #   sample -> reward -> advantage -> assemble
│   ├── eval_workflow.py            #   evaluation sampling + scoring path
│   └── training_workflow.py        #   replay -> backend step -> executor fallback
├── training/                       # Train executor, update schedule, train backends
├── buffer/                         # Buffer subsystem (queue/filter/reassembly/store)
├── distributed/                    # Distributed coordination (for example weight sync)
├── ray/                            # Ray distributed orchestration
│   ├── rollout_manager.py          #   Central orchestrator
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

Then pass it via `--algorithm.algorithm-path your_module.MyAlgorithm`.
See the fully working reference implementation:
`diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm`.
For the current extension contract, see [docs/Algorithm_Minimal_Template.md](docs/Algorithm_Minimal_Template.md).
For the supported user-facing extension surface, see [docs/Supported_Extension_Surface.md](docs/Supported_Extension_Surface.md).

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

- Built-in scorer: set `--reward.reward-model-name <builtin>` for one scorer, or
  `--reward.reward-models a,b --reward.reward-weights wa,wb` for multi-component reward.
- Custom scorer: set `--reward.reward-path your_module.MyRewardScorer` to a full
  scorer class dotpath.
- Execution placement: keep the default manager-local path, set
  `--reward.reward-location sampling_actor` when the sampling host should compute
  rewards inline, or use `--reward.use-http-reward true` plus
  `--reward.reward-service-url ...` to call an HTTP reward service.

Historical compatibility aliases under `diffusionrl.reward.local.*` are removed.
Use the real scorer class path directly.

Subclass `BaseRewardScorer` for the minimal contract:

```python
from diffusionrl.types.reward import RewardRequest, RewardResponse
from diffusionrl.reward.base import BaseRewardScorer

class MyRewardScorer(BaseRewardScorer):
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        # Your reward logic here
        ...
```

Then pass it via `--reward.reward-path your_module.MyRewardScorer`.

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
