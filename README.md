# DiffusionRL

**DiffusionRL** is a distributed reinforcement learning framework for diffusion model optimization, providing two core capabilities:

1.  **High-Performance Distributed Training**: Supports efficient RL training in various deployment modes (separate, colocate, async pipeline) by orchestrating Ray-based inference and training actor pools;
2.  **Flexible Algorithm & Engine Integration**: Enables pluggable RL algorithms (GRPO, MixGRPO, DanceGRPO, FlowGRPO, NFT), rollout engines (trainside / SGLang / vLLM-Omni), and reward models through a unified contract-driven architecture.

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
For real datasets, symlink into `data/datasets/` and override `DATA_PATH` when invoking a `scripts/` wrapper.

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

For external data / model directories, pass absolute paths directly (or create symlinks manually) via `DATA_PATH` / `PRETRAINED_MODEL` when invoking a `scripts/` wrapper.

### Training

The maintained entry point is `python -m diffusionrl.train` with a Hydra
``+experiment=<name>`` selector. Curated launchers under `scripts/` wrap
common runs and inject env-var overrides; the experiment YAMLs under
`conf/experiment/` are the source of truth for recipe geometry.

```bash
# Single-node SD3.5 colocate (vllm-omni + FSDP DP=8 on 1×8 H20)
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate

# Or invoke the entry directly with any +experiment from conf/experiment/
python -m diffusionrl.train \
    +experiment=flowgrpo_fast_sd3_colocate \
    run.data_path=data/samples/prompts_toy.json \
    resume.output_dir=outputs/my_experiment
```

CLI overrides use Hydra dotted-key syntax (e.g.
``run.num_rollouts=100``); see `scripts/run_*.sh` for the full env-var ⇄
Hydra-override pattern that the recipes follow.

Terminology:

- `rollout_id` / `rollout.step` is the outer rollout-train loop step.
- In practice it behaves similarly to a framework-level global step.
- It is not strictly the same as optimizer update count when gradient accumulation or inner epochs are enabled.

Configuration is Hydra-driven. Each experiment YAML under
`conf/experiment/*.yaml` selects a recipe via the `defaults:` list (model,
algorithm, rollout engine, sync variant, etc.) and pins recipe-defining
values. CLI overrides use Hydra's dotted-key syntax
(e.g. ``run.num_rollouts=100``, ``training.plan.local_batch_size=24``);
unknown keys fail fast.

Precision is grouped by runtime owner: ``precision.training.*`` for
training model/FSDP/loss-forward precision; ``precision.rollout.*`` for
rollout autocast plus trajectory/logprob storage precision.

Weight sync is opt-in. Direct-sampling mode
(`rollout/engine: trainside`) forbids a `sync:` section — the train
actor already owns the model. Dedicated-rollout mode
(`rollout/engine: sglang` or `vllm_omni`) requires a sync variant —
typical choices are `tensor_payload` (CUDA-IPC, colocate), `nccl_broadcast`
(separate), or `checkpoint_path` (disk staging). The cross-component
validator enforces this biconditional.

Training geometry is rollout-driven only:

- `algorithm.prompts_per_rollout * algorithm.samples_per_prompt` defines the global rollout batch.
- The resolved local training batch is derived from the training topology.
- `training.num_updates_per_batch` controls how one resolved local batch is split across optimizer updates.
- `training.local_mini_batch_size` is a derived value, computed as `local_batch_size / num_updates_per_batch`.
- `training.micro_batch_size` only controls micro-step slicing inside one local mini-batch and must evenly divide the resolved `local_mini_batch_size`.

### Training Backend Selection

Training composition is policy-driven (no separate "train backend" abstraction).
A typical stack composes ``LoRA → FSDP → EMA`` via
``cfg.training.policies``; the policies are real classes from
`diffusionrl.training` and chain through `PolicyBase.walk_source_chain`.

FSDP2 runtime requires a torch build that provides composable FSDP2 APIs
(`fully_shard`) and distributed checkpoint state-dict helpers.

## Supported Algorithms

| Algorithm | Description | Transition Rule | Key Feature |
|-----------|-------------|----------|-------------|
| **GRPO** | Group Relative Policy Optimization | `flow`, `cps`, `dpm2` | Standard trajectory-based RL |
| **DanceGRPO** | GRPO with dance-specific SDE formulation | `dance` | Optimized for FLUX/SD3 |
| **FlowGRPO** | Flow matching formulation | `flow` | Flow-based objective |
| **MixGRPO** | Mixed ODE/SDE sampling | Configurable | Flexible SDE ratio (0~1) with window scheduler |
| **NFT** | Negative Fine-Tuning | N/A | Forward diffusion, no trajectory needed |

## Configuration Walkthrough

DiffusionRL uses Hydra. Settings group into typed sections under
`cfg.*`; each has a registered dataclass:

| Section | What it owns | Registered under |
|---|---|---|
| `cfg.model` | Model bundle (pretrained ckpt path, dtype, LoRA, model-specific knobs) | `diffusionrl/models/<model>/config.py` (e.g. `model: sd3`, `wan21`) |
| `cfg.algorithm` | Algorithm hyperparams (clip range, schedule, sampling spec) | `diffusionrl/algorithms/` + `rollout_control.py` (`algorithm: grpo`) |
| `cfg.algorithms.<slot>` | Per-segment trainer-side algorithm (e.g. `algorithms.video: DiffusionGRPO`) | Set inline in experiment YAML via `_target_:` |
| `cfg.sampling` | SDE strategy (`flow`, `dance`, `cps`, …), `eta`, `num_inference_steps`, `guidance_scale`, `shift` | `diffusionrl/sde/` + `diffusionrl/config/sampling.py` |
| `cfg.reward` | Reward provider + components + execution mode | `diffusionrl/reward/` (`reward: default`) |
| `cfg.rollout.engine` | Rollout engine (`trainside`, `sglang`, `vllm_omni`) | `diffusionrl/rollout/engine/*/config.py` |
| `cfg.rollout.plan` | Rollout request planning (forward batch size, etc.) | `diffusionrl/rollout/plan.py` |
| `cfg.training.plan` | Training batch geometry (`global_batch_size`, `local_batch_size`, `num_updates_per_batch`, `micro_batch_size`) | `diffusionrl/training/plan.py` |
| `cfg.training.topology` | DP / replica / shard sizes, actor count | `diffusionrl/training/topology.py` |
| `cfg.training.execution` | Offload (train/rollout), gradient accumulation, max-grad-norm | `diffusionrl/training/execution.py` |
| `cfg.training.optimizer`, `cfg.training.lr_scheduler` | Optimizer + LR schedule | `diffusionrl/training/factories.py` |
| `cfg.training.policies` | Composable Policy stack (e.g. `[LoRA, FSDP, EMA]`) | `diffusionrl/training/{lora,fsdp,ema}_policy.py` |
| `cfg.placement` | GPU placement / colocate / sglang split | `diffusionrl/ray/placement.py` |
| `cfg.sync` | Weight-sync variant (only when `rollout.engine` is dedicated) | `diffusionrl/distributed/weight_sync/*` |
| `cfg.run` | Outer loop (num_rollouts, data path, eval cadence) | `diffusionrl/config/run.py` |
| `cfg.logging` | wandb / report destinations | `diffusionrl/config/logging.py` |

To see the resolved cfg for any experiment:

```bash
python -m diffusionrl.train +experiment=<name> --cfg job --resolve
```

## Developer Guide

### Ray Layering

The Ray control plane is split into worker implementations and worker-group orchestration:

- `diffusionrl/ray/{rollout_actor.py, train_actor.py}`: single worker implementations
- `diffusionrl/ray/group/{rollout.py, train.py}`: group orchestration (spawn + dispatch + control plane)
- `diffusionrl/ray/utils/`: stateless helpers imported by actors (`net`, `gpu`, `node`)
- `diffusionrl/rollout/pipeline.py`: driver-side `RolloutPipeline`
- `diffusionrl/rollout/engine/{trainside,sglang,vllm_omni}/`: rollout engine implementations
- `diffusionrl/training/**`: training execution + composable Policy stack
- `diffusionrl/algorithms/**`: stage-driven RL algorithms (GRPO, ARGRPO, GRPORolloutControl)
- `diffusionrl/distributed/weight_sync/**`: trainer→rollout weight broadcast (NCCL, IPC, tensor payload, checkpoint)

### Project Structure

```
diffusionrl/
├── train.py                    # Training entry point
├── types/                          # Canonical shared data types (RolloutRequest, RolloutResp, TrajectoryStore, segments, etc.)
├── config/                         # Hydra registration + validation + instantiation
├── algorithms/                     # Stage-driven algorithms (DiffusionGRPO, ARGRPO, GRPORolloutControl, normalizers)
├── models/                         # Per-model stack (sd3, wan21, wan22, qwen_image, hunyuan_video15, hunyuan_image3)
├── reward/                         # Reward executors + scorers (local + remote service)
├── data/                           # Data loading and datasets
├── rollout/                        # Driver rollout runtime
│   ├── pipeline.py                 #   RolloutPipeline
│   └── engine/{trainside,sglang,vllm_omni}/
├── training/                       # Stage train stack + composable Policy chain (FSDP, LoRA, EMA)
├── buffer/                         # Buffer subsystem (queue/filter/store)
├── distributed/                    # Distributed coordination (weight sync, transfer queue)
├── ray/                            # Ray distributed orchestration
│   ├── rollout_actor.py / train_actor.py
│   ├── group/                      #   rollout.py / train.py (group orchestration)
│   ├── mixins/                     #   reusable actor mixins
│   ├── buffer_actor.py / placement.py / distributed.py
│   └── utils/                      #   net/gpu/node stateless helpers
├── sde/                            # SDE runtime (sigma schedules, step indices)
├── patches/                        # Runtime patches
└── utils/                          # Checkpointing, logging, media helpers
```

### Adding a Custom Algorithm

Subclass `diffusionrl.algorithms.base.StageAlgorithm` and implement
`compute_loss_and_backward(*, conditions, segment, advantages, training_progress, loss_scale)`,
returning an `AlgorithmStepResult`. The stage (e.g. `DiffusionStage`,
`ARStage`) owns the forward path; the algorithm is pure loss math against
`segment.sde_logp` / `segment.log_probs`. Reference implementations:
`diffusionrl/algorithms/grpo.py` (`DiffusionGRPO`, `ARGRPO`).

For driver-side rollout control (advantage normalization, SDE-index
scheduling, training-index filtering), subclass `GRPORolloutControl` in
`diffusionrl/algorithms/rollout_control.py`.

Wire your algorithm into an experiment via Hydra:

```yaml
# conf/experiment/my_recipe.yaml
algorithms:
  video:
    _target_: my_module.MyDiffusionAlgo
```

### Plugin Templates (diffusionrl_plugins)

`diffusionrl_plugins/` is a third-party extension namespace. Currently the
maintained example is the reward template:

- Reward: `diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer`

There is no plugin auto-registration. Reference your extension via Hydra
`_target_:` in an experiment YAML.

### Adding a Custom Reward Scorer

Subclass `BaseRewardScorer` and wire it into the experiment via Hydra
`_target_:`:

```python
from diffusionrl.types.reward import RewardRequest, RewardResponse
from diffusionrl.reward.base import BaseRewardScorer

class MyRewardScorer(BaseRewardScorer):
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        # Your reward logic here
        ...
```

```yaml
# conf/experiment/my_recipe.yaml
reward:
  provider:
    _target_: my_module.MyRewardScorer
```

If you want the built-in local scorer lifecycle helpers (`device`, eager
model load, standard `offload()` / `onload()` behavior), subclass
`diffusionrl.reward.scorers.base_local.BaseLocalRewardScorer` instead.

### Running Tests

```bash
# Hydra compose check (replace +experiment= with any conf/experiment/*.yaml)
python -m diffusionrl.train +experiment=flowgrpo_fast_sd3_colocate --cfg job --resolve

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
