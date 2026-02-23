# DiffusionRL

**DiffusionRL** is a distributed reinforcement learning framework for diffusion model optimization, providing two core capabilities:

1.  **High-Performance Distributed Training**: Supports efficient RL training in various deployment modes (separate, colocate, async pipeline) by orchestrating Ray-based inference and training actor pools;
2.  **Flexible Algorithm & Engine Integration**: Enables pluggable RL algorithms (GRPO, MixGRPO, DanceGRPO, FlowGRPO, NFT), sampling engines (FSDP, FastVideo, SGLang), and reward models through a unified contract-driven architecture.

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

- **Training Actors (Ray + FSDP)**: Responsible for the main training process, reads `TrainingBatch` from the rollout pipeline, and synchronizes parameters to the inference actors after training.
- **Inference Actors (FSDP / FastVideo / SGLang)**: Generates denoising trajectories and latent samples using pluggable sampling engines, producing `SamplerOutput` with a unified v1 contract.
- **Reward Service**: Evaluates generated samples using configurable reward models (HPS, CLIP, PickScore, OCR, etc.) and feeds reward signals back to the training loop.
- **RolloutManager**: The central orchestrator that coordinates sampling, reward computation, advantage calculation, and batch assembly.

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      RolloutManager                         │
 │  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
 │  │  Prompts  │───>│  Inference   │───>│  Reward Service  │  │
 │  │  (Data)   │    │  ActorGroup  │    │  (Local / HTTP)  │  │
 │  └──────────┘    └──────┬───────┘    └────────┬─────────┘  │
 │                         │ SamplerOutput (v1)   │ rewards    │
 │                         v                      v            │
 │               ┌─────────────────────────────────┐           │
 │               │  Advantage Calc + Batch Assembly │           │
 │               └──────────────┬──────────────────┘           │
 └──────────────────────────────┼──────────────────────────────┘
                                │ TrainingBatch
                                v
                    ┌───────────────────────┐
                    │   Training ActorGroup  │
                    │   (FSDP + LoRA/Full)   │
                    └───────────┬───────────┘
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

```bash
# Option A (recommended): editable install with core extras
pip install -e ".[train,infer,eval]"

# Option B: requirements-based install (same dependency set as Option A)
pip install -r requirements.txt
pip install -e . --no-deps
```

To match our validated GPU environment, install FlashAttention explicitly:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For development tools:

```bash
pip install -e ".[train,infer,eval,dev]"
```

If your package mirror blocks build-time `setuptools/wheel` resolution, use:

```bash
pip install -e ".[train,infer,eval]" --no-build-isolation
# or
pip install -r requirements.txt --no-build-isolation
pip install -e . --no-deps --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`mmcv` / `mmdetection` are intentionally **not** installed by default.
Install them only when needed (Geneval/OpenMMLab workflows):

- [docs/geneval_mmcv_setup.md](docs/geneval_mmcv_setup.md)

### Data Preparation

Toy data is included in `data/samples/` for smoke tests.
For real datasets, symlink into `data/datasets/` and override `DATA_PATH`:

```bash
DATA_PATH=data/datasets/hpdv2/train.json bash scripts/train_dancegrpo_sd3_separate.sh
```

Default local paths are resolved against the repository root:

- `data_path`: `data/samples/prompts_toy.json`
- `output_dir`: `outputs/`
- `weight_sync_dir`: `outputs/weight_sync/`
- local model mount: `models/local/`

For external data / model directories, use the linking script:

```bash
bash scripts/link_external_resources.sh \
    --data-source /path/to/external/data \
    --models-source /path/to/external/shared_models
```

### Training

We provide pre-configured training scripts for various algorithm + model combinations:

```bash
# DanceGRPO with FLUX (separate mode, 4+4 GPUs)
bash scripts/train_dancegrpo_flux_separate.sh

# MixGRPO with SD3 (separate mode)
bash scripts/train_mixgrpo_sd3_separate.sh

# NFT with SD3 (separate mode)
bash scripts/train_nft_sd3_separate.sh

# Override default parameters
bash scripts/train_dancegrpo_flux_separate.sh \
    --num-rollout 100 \
    --batch-size 2 \
    --inference-gpus 4 \
    --training-gpus 4
```

Or use the CLI directly:

```bash
python -m diffusionrl.train \
    --pretrained-model-saved-path black-forest-labs/FLUX.1-dev \
    --model-type flux \
    --sampler-path diffusionrl.samplers.fsdp.flux_sampler.FluxSampler \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name hpsv2 \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path data/samples/prompts_toy.json \
    --sde-type flux_dance \
    --eta 0.3 \
    --num-inference-steps 25 \
    --use-lora true \
    --lora-rank 16 \
    --num-rollout 300 \
    --output-dir outputs/my_experiment
```

### Available Training Scripts

| Script | Algorithm | Model | Mode |
|--------|-----------|-------|------|
| `train_dancegrpo_flux_separate.sh` | DanceGRPO | FLUX | Separate |
| `train_dancegrpo_sd3_separate.sh` | DanceGRPO | SD3 | Separate |
| `train_dancegrpo_hunyuan_separate.sh` | DanceGRPO | Hunyuan | Separate |
| `train_dancegrpo_hunyuan_fastvideo_*.sh` | DanceGRPO | Hunyuan | Separate / Colocate |
| `train_mixgrpo_flux_separate.sh` | MixGRPO | FLUX | Separate |
| `train_mixgrpo_sd3_separate.sh` | MixGRPO | SD3 | Separate |
| `train_flowgrpo_sd3_separate.sh` | FlowGRPO | SD3 | Separate |
| `train_nft_sd3_separate.sh` | NFT | SD3 | Separate |

Each script also has a `*_train_actor_sampling.sh` variant that uses training actors for sampling.

## Supported Algorithms

| Algorithm | Description | SDE Type | Key Feature |
|-----------|-------------|----------|-------------|
| **GRPO** | Group Relative Policy Optimization | `sde`, `cps`, `dpm2` | Standard trajectory-based RL |
| **DanceGRPO** | GRPO with dance-specific SDE formulation | `dance`, `flux_dance` | Optimized for FLUX/SD3 |
| **FlowGRPO** | Flow matching formulation | `flux_flow` | Flow-based objective |
| **MixGRPO** | Mixed ODE/SDE sampling | Configurable | Flexible SDE ratio (0~1) with window scheduler |
| **NFT** | Noise-Free Training | N/A | Forward diffusion, no trajectory needed |

## Arguments Walkthrough

Arguments in DiffusionRL are organized into the following categories:

1.  **Model arguments**: `--model-type`, `--pretrained-model-saved-path`, `--use-lora`, `--lora-rank`, `--lora-alpha`, etc.
2.  **Sampling arguments**: `--sde-type`, `--eta`, `--num-inference-steps`, `--guidance-scale`, `--shift`, `--timestep-fraction`, etc.
3.  **Algorithm arguments**: `--algorithm-path`, `--clip-range`, `--use-kl-penalty`, `--advantage-type`, etc.
4.  **Reward arguments**: `--reward-path`, `--reward-model-name`, `--reward-batch-size`, etc.
5.  **Training arguments**: `--learning-rate`, `--gradient-accumulation-steps`, `--max-grad-norm`, `--num-inner-epochs`, etc.
6.  **Runtime arguments**: `--colocate-inference-training`, `--inference-num-gpus-per-node`, `--training-num-gpus-per-node`, `--placement-strategy`, etc.

For the full argument reference, please refer to: [diffusionrl/config/arguments.py](diffusionrl/config/arguments.py)

## Developer Guide

### Ray Layering

The Ray control plane is split into worker implementations, worker-group orchestration, and train-loop strategy:

- `diffusionrl/ray/actors/`: single worker implementations (`InferenceActor`, `TrainingActor`)
- `diffusionrl/ray/groups/`: group orchestration (`BaseActorGroup`, `InferenceActorGroup`, `TrainingActorGroup`, factories)
- `diffusionrl/ray/sampling_mode/`: train-loop strategy plugins (`inference` / `training` backend transitions)
- `diffusionrl/ray/rollout_manager.py`: control-plane actor
- `diffusionrl/runtime/**`: Ray-agnostic runtime logic

Detailed layer diagram and migration notes:
- [docs/Ray_Layering.md](docs/Ray_Layering.md)
- Legacy `actor_group_*` import paths are kept as compatibility shims for one release cycle.

### Project Structure

```
diffusionrl/
├── train.py / train_async.py      # Training entry points
├── types/                          # Canonical shared data types (SamplerOutput, TrainingBatch, Reward, WeightSync)
├── config/                         # Configuration system (TrainingArguments)
├── algorithms/                     # RL algorithms (GRPO, MixGRPO, NFT)
├── samplers/                       # Inference engines (FSDP, FastVideo, SGLang)
│   ├── fsdp/                       #   FSDP-based: FluxSampler, SD3Sampler, HunyuanSampler
│   ├── fastvideo/                  #   FastVideo-based: FastVideoSampler
│   └── sglang/                     #   SGLang external service engine
├── losses/                         # Loss functions (GRPOLoss, NFTLoss)
├── advantages/                     # Advantage computation (global, group, per-prompt)
├── reward/                 # Reward workers (Local, HTTP, Ray service)
├── models/                         # Model implementations (FLUX, SD3, Hunyuan, Mochi)
├── data/                           # Data loading and datasets
├── ray/                            # Ray distributed orchestration
│   ├── rollout_manager.py          #   Central orchestrator
│   ├── actors/                     #   Worker implementations
│   ├── groups/                     #   Worker-group orchestration
│   └── sampling_mode/              #   Mode strategies in train loop
├── runtime/                        # Async runtime + ray-agnostic execution logic
├── patches/                        # Non-invasive patches for FastVideo
└── utils/                          # Checkpointing, logging, EMA, weight sync
```

### Adding a Custom Algorithm

Subclass `BaseAlgorithm` and define your `SamplingRequirements`.
For new code, always import shared data types from `diffusionrl.types` (single entry).

```python
from diffusionrl.algorithms.base import BaseAlgorithm, SamplingRequirements
from diffusionrl.types import BackwardTrainingBatch

class MyAlgorithm(BaseAlgorithm):
    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            sde_ratio=1.0,
        )

    def compute_loss(self, model, batch, timestep_idx, advantages, **kwargs):
        # Optional typed path if your algorithm uses typed batch data
        if isinstance(batch, BackwardTrainingBatch):
            ...
        ...
```

Then pass it via `--algorithm-path your_module.MyAlgorithm`.
See the full minimal template: [docs/Algorithm_Minimal_Template.md](docs/Algorithm_Minimal_Template.md)

### Adding a Custom Reward Worker

Subclass `BaseRewardWorker`:

```python
from diffusionrl.reward.base import BaseRewardWorker

class MyRewardWorker(BaseRewardWorker):
    def compute_reward(self, images, prompts, **kwargs):
        # Your reward logic here
        return rewards
```

Then pass it via `--reward-path your_module.MyRewardWorker`.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_model_bundle_inference_hooks.py -v

# With coverage
pytest tests/ --cov=diffusionrl --cov-report=term-missing
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
