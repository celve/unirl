<div align="center">

# UniRL

### A Reinforcement Learning Framework for Unified Multimodal Models

**U**(you)·**ni**(need)·**RL** for unified multimodal intelligence

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-unirl--project.github.io-blue)](https://unirl-project.github.io/unirl/)
[![WeChat](https://img.shields.io/badge/WeChat-微信群-07C160?logo=wechat&logoColor=white)](assets/wechat_qr.jpg)

</div>

## News 🚀

- **[2026-05]** **DRPO** released — *"Rethinking the Divergence Regularization in LLM Reinforcement Learning"* ([arXiv]()).
- **[2026-05]** **FlowDPPO** released — *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* ([arXiv]()).

## About 💡

UniRL applies one RL post-training loop — generate samples, score them, compute
advantages, update the policy, and optionally sync weights back to rollout workers —
across multimodal model families, instead of binding it to one model type or rollout
backend. The runtime pieces are all first-class and composable:

- **Unified multimodal RL loop.** One trainer pattern covers diffusion, AR,
  prompt-enhancer, and mixed AR + diffusion models.
- **Flexible rollout engines.** Train-side sampling, SGLang, SGLang LLM,
  vLLM-Omni, and composed rollout backends.
- **Distributed by design.** Ray placement, FSDP workers, rollout pools,
  offload/onload, and weight sync are first-class runtime pieces.
- **Example-first experiments.** Hydra example configs define models, algorithms,
  rollout, rewards, placement, sync, and batch geometry — examples under
  `examples/` are the source of truth.
- **Extensible model packages.** Model-specific bundles, pipelines, conditions,
  AR/diffusion logic, and VAE code live behind shared contracts.

## Algorithms

### 🌟 Team-Proposed Algorithms

> **🌟 These algorithms are proposed by our team — the highlight of UniRL.** Each
> algorithm's folder holds a step-by-step tutorial, a runnable example recipe, and
> (where available) a released checkpoint. We highly recommend trying them in our framework!

| Algorithm | Paper | Tutorial | Notes |
|---|---|---|---|
| **FlowDPPO** | *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* | [FlowDPPO/](FlowDPPO/) | Diffusion/flow RL with an exact Gaussian-KL trust-region mask. |
| **DRPO** | *"Rethinking the Divergence Regularization in LLM Reinforcement Learning"* | [DRPO/](DRPO/) | Token-level AR/LLM RL with a smooth Binary-TV quadratic regularizer. |

### Public Algorithms

Public/reference algorithms currently wired into UniRL examples and training code.

| Algorithm | Coverage | Notes |
|---|---|---|
| GRPO / ARGRPO | `examples/diffusion/`, `examples/vlm/`, `examples/pe/`, `examples/unified_model/`; `unirl/algorithms/{diffusion_grpo,ar_grpo}.py` | Group-relative PPO objective for diffusion and AR stages. |
| NFT | `examples/diffusion/`; `unirl/algorithms/nft.py` | Forward-process diffusion fine-tuning. |
| DanceGRPO | `examples/diffusion/`; `unirl/algorithms/diffusion_grpo.py`, `unirl/sde/kernels.py` | DiffusionGRPO with Dance SDE settings. |
| MixGRPO | `examples/diffusion/`; `unirl/algorithms/diffusion_grpo.py`, `unirl/utils/scheduler_utils.py` | DiffusionGRPO with mixed/windowed timestep scheduling. |

## Model Support 🎨

Model and algorithm support are **two independent dimensions** that compose within
a domain: any diffusion algorithm (see [Algorithms](#algorithms)) runs on a diffusion
model, AR algorithms on AR models — so UniRL covers many more model × algorithm
combinations than the shipped example recipes alone. The table below is the model
dimension; maturity is ✅ stable · 🧪 experimental (1–2 recipes).

| Model | Category | Modality | Status |
|---|---|---|---|
| Stable Diffusion 3 | Image diffusion | Text → Image | ✅ |
| Qwen-Image | Image diffusion | Text → Image | ✅ |
| FLUX.2-Klein | Image diffusion | Text → Image | 🧪 |
| WAN 2.1 | Video diffusion | Text / Image → Video | ✅ |
| WAN 2.2 | Video diffusion | Text / Image → Video | ✅ |
| HunyuanVideo | Video diffusion | Text → Video | 🧪 |
| HunyuanVideo 1.5 | Video diffusion | Text → Video | 🧪 |
| Qwen-VL | Vision-language AR | Text + Image → Text | ✅ |
| Qwen3 | LLM AR | Text → Text | ✅ |
| Prompt-enhancer | AR prompt rewriter | Text → Text → Image | ✅ |
| HunyuanImage3 | Unified AR + diffusion | Text → Image | ✅ |

Select a model's example with `--config-name=<domain>/<model>/<example>` and launch
it through the matching entrypoint (`train_diffusion`, `train_vlm`, `train_pe`,
or `train_unified_model`). For example:

```bash
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside
```

## Getting Started 🚀

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
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
```

Select another domain entrypoint with `ENTRY`:

```bash
ENTRY=train_vlm bash examples/run_experiment_single_node.sh vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8
ENTRY=train_pe  bash examples/run_experiment_single_node.sh pe/pe/pe_trainside_pickscore
```

Invoke an entrypoint directly when you do not need the shell launchers:

```bash
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8
```

## Examples 📂

UniRL covers five training modes, one Hydra example bucket and entrypoint each.
Examples are self-contained YAML files selected with
`--config-name=<domain>/<model>/<example>`:

| Domain | Trains | Entrypoint | Example |
|---|---|---|---|
| `diffusion/` | Image / video diffusion models | `train_diffusion` | `diffusion/sd3/sd3_sglang_native_colocate` |
| `vlm/` | Vision-language autoregressive (VLM) models | `train_vlm` | `vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8` |
| `llm/` | Text-only autoregressive (LLM) models | `train_vlm` | `llm/qwen3/ar_drpo_qwen3_4b_base_dpao_sglang` |
| `pe/` | Prompt-enhancer (AR rewriter + diffusion reward) | `train_pe` | `pe/pe/pe_sglang_full_pickscore` |
| `unified_model/` | Unified AR + diffusion models | `train_unified_model` | `unified_model/hi3/hi3_vllmomni` |

Every example starts with `# @package _global_`, so its keys compose at the Hydra
config root. For layout responsibilities, naming conventions, and the process for
adding an example, read `examples/README.md`.

## Pipeline 🔁

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

## Roadmap 🗺️

We are actively expanding model and algorithm coverage. Near-term directions:

- Promote the experimental (🧪) models — FLUX.2-Klein, HunyuanVideo, and
  HunyuanVideo 1.5 — to full algorithm coverage.
- Extend the team-proposed algorithms (FlowDPPO, DRPO) to more model families and
  release additional checkpoints.
- Broaden reward backends and rollout-engine coverage across domains.

Want a model or algorithm prioritized? [Open an issue](https://github.com/haonan3/UniRL/issues) to discuss.

## Contributing 🤝

Contributions and questions are welcome. Before opening a pull request, read the
repository conventions in [`AGENTS.md`](AGENTS.md), run the
[development checks](examples/README.md#development-checks) for the files you
touched, and fill in the [pull request template](.github/pull_request_template.md).
For questions, bug reports, and feature requests,
[open an issue](https://github.com/haonan3/UniRL/issues).

## Acknowledgement 🙏

UniRL builds on ideas and infrastructure from the open-source RL and inference
ecosystem. We especially thank
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[slime](https://github.com/THUDM/slime), and
[verl](https://github.com/volcengine/verl).

## Citation 📚

If you find UniRL helpful, please cite:

```bibtex
@misc{unirl_github,
  title        = {{UniRL: Unified Reinforcement Learning for Unified Models}},
  author       = {Haonan Wang and Linyu Wu and Qian Qiu and Lewei Jin and Bowen Ping and Jianghai Chen and Yiheng Du and Guangxin He and Yu Shi and Yongguang Lin and Zhuoxin Zhou and Zhanchao Zhou and Keming Wu and Rizhen Hu and Xuefei Ning and Feiyu Hu and Xiangyan Liu and Siqi Kou and Jiarui Yao and Xiangxin Zhou and Liefeng Bo and Wenxi Zhu and Tianyu Pang},
  year         = {2026},
  howpublished = {\url{https://github.com/Tencent-Hunyuan/UniRL}},
  urldate      = {2026-06-05}
}
```
