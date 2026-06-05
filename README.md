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

<div align="center">
  <img src="assets/UniRL_arch.png" alt="UniRL architecture" width="900">
</div>

UniRL is a layered, composable system. Each **entrypoint** (`train_diffusion`,
`train_vlm`, `train_pe`, `train_unified_model`) loads a **Hydra example config**
covering model, algorithm, rollout, reward, placement, and sync, then creates the
matching domain **trainer** (`DiffusionTrainer`, `VLMTrainer`, `PETrainer`,
`UnifiedModelTrainer`). The trainer coordinates the RL loop across pluggable
**rollout engines**, **algorithms**, **model bundles**, **reward services**, and
the shared **distributed runtime**: Ray `DevicePool`, FSDP, placement, transfer
queue, and LoRA/full-weight sync. See [`unirl/README.md`](unirl/README.md) for the
runtime loop, deployment modes, and module map.

## Team-Proposed Algorithms 🌟

> **🌟 These algorithms are proposed by our team — the highlight of UniRL.** Each
> algorithm's folder holds a step-by-step tutorial, a runnable example recipe, and
> (where available) a released checkpoint. We highly recommend trying them in our framework!

| Algorithm | Paper | Tutorial | Notes |
|---|---|---|---|
| **FlowDPPO** | *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models"* | [FlowDPPO/](FlowDPPO/) | Diffusion/flow RL with an exact Gaussian-KL trust-region mask. |
| **DRPO** | *"Rethinking the Divergence Regularization in LLM Reinforcement Learning"* | [DRPO/](DRPO/) | Token-level AR/LLM RL with a smooth Binary-TV quadratic regularizer. |

UniRL also wires in standard reference algorithms — **(AR)GRPO**, **DiffusionNFT**,
**DanceGRPO**, and **MixGRPO** — in [`unirl/algorithms/`](unirl/algorithms/README.md).

## Model Support 🎨

Model and algorithm support are **two independent dimensions** that compose within
a domain: any diffusion algorithm (see above) runs on a diffusion
model, AR algorithms on AR models — so UniRL covers many more model × algorithm
combinations than the shipped example recipes alone. The table below is the model
dimension; maturity is ✅ stable · 🧪 experimental (1–2 recipes).

| Model | Category | Modality | Status |
|---|---|---|---|
| Stable Diffusion 3 / 3.5 | Image diffusion | Text → Image | ✅ |
| Qwen-Image | Image diffusion | Text → Image | ✅ |
| FLUX.2-Klein | Image diffusion | Text → Image | 🧪 |
| WAN 2.1 | Video diffusion | Text / Image → Video | ✅ |
| WAN 2.2 | Video diffusion | Text / Image → Video | ✅ |
| HunyuanVideo 1.0 / 1.5 | Video diffusion | Text → Video | 🧪 |
| Qwen-VL | Vision-language AR | Text + Image → Text | ✅ |
| Qwen3 | LLM AR | Text → Text | ✅ |
| Prompt-enhancer | LLM + diffusion | Text → Text → Image | ✅ |
| HunyuanImage3 | Unified AR + diffusion | Text → Image | ✅ |

Each model maps to a domain entrypoint (`train_diffusion`, `train_vlm`, `train_pe`,
`train_unified_model`); see **Getting Started** below to run any of them.

## Training Modes 🧩

UniRL unifies five training modes, one Hydra example bucket and entrypoint each.
Examples are self-contained YAML files selected with
`--config-name=<domain>/<example>`:

| Domain | Trains | Entrypoint | Example |
|---|---|---|---|
| `diffusion/` | Image / video diffusion models | `train_diffusion` | `diffusion/sd3_sglang_native_colocate` |
| `vlm/` | Vision-language autoregressive (VLM) models | `train_vlm` | `vlm/qwen_vl_argrpo_geo3k_mc_4x8` |
| `llm/` | Text-only autoregressive (LLM) models | `train_vlm` | `llm/qwen3_ar_drpo_4b_base_dpao_sglang` |
| `pe/` | Prompt-enhancer (AR rewriter + diffusion reward) | `train_pe` | `pe/pe_sglang_full_pickscore` |
| `unified_model/` | Unified AR + diffusion models | `train_unified_model` | `unified_model/hi3_vllmomni` |

See [`examples/README.md`](examples/README.md) for the full launch guide, naming
schema, and how to add a recipe.

## Getting Started ⚡

```bash
# install (full guide: INSTALL.md)
pip install -e ".[train,infer,eval]" --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation

# compose-check, then launch a single-node example
python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside --cfg job --resolve
bash examples/run_experiment_single_node.sh diffusion/sd3_trainside
```

- Full [installation guide](INSTALL.md) — dev tools, environment variables.
- Full [launch guide](examples/README.md#running-a-recipe) — multi-node, every entrypoint, mooncake.

## Roadmap 🗺️

We are actively expanding model and algorithm coverage. Near-term directions:

- Promote the experimental (🧪) models — FLUX.2-Klein and HunyuanVideo 1.0 / 1.5 —
  to full algorithm coverage.
- Extend the team-proposed algorithms (FlowDPPO, DRPO) to more model families and
  release additional checkpoints.
- Broaden reward backends and rollout-engine coverage across domains.

Want a model or algorithm prioritized? [Open an issue](https://github.com/haonan3/UniRL/issues) to discuss.

## Contributing 🤝

Contributions and questions are welcome. Before opening a pull request, read the
repository conventions in [`AGENTS.md`](AGENTS.md), run the
[pre-PR checks](examples/README.md#adding-or-editing-a-recipe) for the files you
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
  title        = {{UniRL: A Reinforcement Learning Framework for Unified Multimodal Models}},
  author       = {Haonan Wang and Linyu Wu and Qian Qiu and Lewei Jin and Bowen Ping and Jianghai Chen and Yiheng Du and Guangxin He and Yu Shi and Yongguang Lin and Zhuoxin Zhou and Zhanchao Zhou and Keming Wu and Rizhen Hu and Xuefei Ning and Feiyu Hu and Xiangyan Liu and Siqi Kou and Jiarui Yao and Xiangxin Zhou and Liefeng Bo and Wenxi Zhu and Tianyu Pang},
  year         = {2026},
  howpublished = {\url{https://github.com/Tencent-Hunyuan/UniRL}},
  urldate      = {2026-06-05}
}
```
