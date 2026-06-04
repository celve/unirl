# Unified Reinforcement Learning Framework

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/haonan3/UniRL?style=social)](https://github.com/haonan3/UniRL/stargazers)
[![Documentation](https://img.shields.io/badge/docs-README-blue)](#getting-started)
[![Open Issues](https://img.shields.io/github/issues/haonan3/UniRL)](https://github.com/haonan3/UniRL/issues)

UniRL is a reinforcement learning framework for diffusion, autoregressive,
prompt-enhancer, and unified models.

[Getting Started](#getting-started) |
[Examples](#examples) |
[Algorithms](#algorithms) |
[Pipeline](#pipeline) |
[Development Checks](#development-checks) |
[Contact Us](#contact-us) |
[Acknowledgement](#acknowledgement)

## News

- **[2026-05]** **DRPO** released — *"Rethinking the Divergence Regularization in LLM Reinforcement Learning"* ([arXiv]()).
- **[2026-05]** **FlowDPPO** released — *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* ([arXiv]()).

## About

UniRL follows the same high-level post-training pattern used by modern RL systems:
generate samples, score them, compute advantages, update the policy, and optionally
sync weights back to rollout workers. The framework applies that loop across
multimodal model families instead of binding it to one model type or rollout
backend.

The core code is organized around typed model packages, Hydra example configs,
Ray worker groups, FSDP train stacks, pluggable rollout engines, and reusable
stage-level loss algorithms. Examples under `examples/` are the source of truth
for each experiment.

## Key Features

- **Unified multimodal RL loop.** One trainer pattern covers diffusion, AR,
  prompt-enhancer, and mixed AR + diffusion models.
- **Flexible rollout engines.** Supports train-side sampling, SGLang, SGLang
  LLM, vLLM-Omni, and composed rollout backends.
- **Distributed by design.** Ray placement, FSDP workers, rollout pools,
  offload/onload, and weight sync are first-class runtime pieces.
- **Example-first experiments.** Hydra example configs define models, algorithms,
  rollout, rewards, placement, sync, and batch geometry.
- **Extensible model packages.** Model-specific bundles, pipelines,
  conditions, AR/diffusion logic, and VAE code live behind shared contracts.

## Supported Capabilities

| Area | Example domain | Model dirs / current support |
|---|---|---|
| Image/video diffusion RL | `diffusion/` | `sd3/`, `qwen_image/`, `flux2_klein/`, `wan21/`, `wan22/`, `hunyuan_video/`, `hunyuan_video15/`; includes GRPO-style, FlowDPPO, NFT, DanceGRPO, and MixGRPO example configs where available. |
| Vision-language AR RL | `vlm/` | `qwen_vl/`; Qwen-VL ARGRPO examples on Geo3K-style multiple-choice data, including SGLang and LoRA variants. |
| Text-only AR RL | `llm/` | `qwen3/`; Qwen3 DRPO example coverage with SGLang rollout. |
| Prompt-enhancer RL | `pe/` | `pe/`; train-side and SGLang prompt-enhancer examples with full/LoRA variants and PickScore/WISE reward choices. |
| Mixed AR + diffusion RL | `unified_model/` | `hi3/`; HunyuanImage3 unified-model examples with vLLM-Omni rollout. |

Select examples with `--config-name=<domain>/<model>/<example>` and launch
them through the matching entrypoint (`train_diffusion`, `train_vlm`, `train_pe`,
or `train_unified_model`). For example:

```bash
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside
```

## Algorithms

### Team-Proposed Algorithms

This section highlights algorithms proposed by our team. You are recommended to try them in our framework!

| Algorithm | Paper | Tutorial / example | Notes |
|---|---|---|---|
| FlowDPPO | *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* | `FlowDPPO/`, `examples/diffusion/sd3/sd3_flowdppo.yaml` | Diffusion/flow RL with an exact Gaussian-KL trust-region mask. |
| DRPO | *"Rethinking the Divergence Regularization in LLM Reinforcement Learning"* | `DRPO/`, `examples/llm/qwen3/ar_drpo_qwen3_4b_base_dpao_sglang.yaml` | Token-level AR/LLM RL with a smooth Binary-TV quadratic regularizer. |

### Public Algorithms

Public/reference algorithms currently wired into UniRL examples and training code.

| Algorithm | Coverage | Notes |
|---|---|---|
| GRPO / ARGRPO | `examples/diffusion/`, `examples/vlm/`, `examples/pe/`, `examples/unified_model/`; `unirl/algorithms/{diffusion_grpo,ar_grpo}.py` | Group-relative PPO objective for diffusion and AR stages. |
| NFT | `examples/diffusion/`; `unirl/algorithms/nft.py` | Forward-process diffusion fine-tuning. |
| DanceGRPO | `examples/diffusion/`; `unirl/algorithms/diffusion_grpo.py`, `unirl/sde/kernels.py` | DiffusionGRPO with Dance SDE settings. |
| MixGRPO | `examples/diffusion/`; `unirl/algorithms/diffusion_grpo.py`, `unirl/utils/scheduler_utils.py` | DiffusionGRPO with mixed/windowed timestep scheduling. |

## Getting Started

### Install

Install the core training, inference, and evaluation dependencies:

```bash
pip install -e ".[train,infer,eval]" --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For development tools:

```bash
pip install -e ".[train,infer,eval,dev]" --no-build-isolation
```

Examples read cluster-local paths, checkpoints, data, and W&B settings from
environment variables through `${oc.env:...}`. Common variables include
`PRETRAINED_MODEL`, `DATA_PATH`, `EVAL_DATA_PATH`, `REPORT_TO_WANDB`,
`WANDB_PROJECT`, and `WANDB_ENTITY`. Small prompt lists are committed under
`datasets/`.

### Quick Start

Compose and resolve an example before launching a run:

```bash
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve
```

Launch a single-node diffusion example:

```bash
bash scripts/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
```

Select another domain entrypoint with `ENTRY`:

```bash
ENTRY=train_vlm bash scripts/run_experiment_single_node.sh vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8
ENTRY=train_pe  bash scripts/run_experiment_single_node.sh pe/pe/pe_trainside_pickscore
```

Invoke an entrypoint directly when you do not need the shell launchers:

```bash
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8
```

## Examples

Examples are self-contained YAML files selected with
`--config-name=<domain>/<model>/<example>`:

| Domain | Example |
|---|---|
| `diffusion/` | `diffusion/sd3/sd3_sglang_native_colocate` |
| `vlm/` | `vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8` |
| `llm/` | `llm/qwen3/ar_drpo_qwen3_4b_base_dpao_sglang` |
| `pe/` | `pe/pe/pe_sglang_full_pickscore` |
| `unified_model/` | `unified_model/hi3/hi3_vllmomni` |

Every example starts with `# @package _global_`, so its keys compose at the Hydra
config root. For layout responsibilities, naming conventions, and the process for
adding an example, read `examples/README.md`.

## Pipeline

```text
prompts
  -> rollout workers
       trainside | sglang | sglang_llm | vllm_omni | composed
  -> rewards
  -> advantages
  -> train workers
       model bundle + FSDP train stack + stage algorithm
  -> optional weight sync
       lora/full weights over nccl | tensor | ipc
```

At startup, an entrypoint composes the selected example, validates cross-component
contracts, builds a domain trainer, acquires a Ray `DevicePool`, and constructs
rollout and train workers. The trainer then runs the rollout -> reward ->
advantage -> train -> optional weight-sync loop.

Deployment mode is controlled by the rollout engine and optional `sync:` section:

| Mode | Shape | Sync requirement |
|---|---|---|
| Train-side sampling | Train workers also sample | No `sync:` section |
| Separate rollout | Rollout and training use different GPU pools | Required |
| Colocated rollout | Rollout and training share GPU bundles with offload/onload | Required |

## Development Checks

Before submitting a change, run the checks
that match the files you touched:

```bash
# Compose one changed or representative example and print the resolved config
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve

# Python syntax check
python -m compileall -q unirl

# Shell launcher syntax check
for f in scripts/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```

## Contact Us

For questions, bug reports, and feature requests, please open an issue at
[haonan3/UniRL](https://github.com/haonan3/UniRL/issues).

## Acknowledgement

UniRL builds on ideas and infrastructure from the open-source RL and inference
ecosystem. We especially thank
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[slime](https://github.com/THUDM/slime), and
[verl](https://github.com/volcengine/verl).
