# UniRL

UniRL is a distributed reinforcement learning framework for unified multimodal
generative models. It trains diffusion and autoregressive models with Ray-based
worker groups, Hydra experiment recipes, composable training stacks, and
pluggable rollout engines.

```text
   prompts ──▶ rollout workers ──▶ rewards ──▶ advantages ──┐
                  (sample x_T → x_0)                         │
                                                             ▼
                                                      train workers
                                                      (FSDP stack +
                                                       loss algorithm)
                                                             │
                                       weight sync ◀─────────┘
                                       (dedicated rollout only)
```

| Domain | Models | Algorithms |
|---|---|---|
| Image | Stable Diffusion 3, Qwen-Image, HunyuanImage3 | GRPO, DanceGRPO, MixGRPO, Flow-DPPO, NFT |
| Video | WAN 2.1 / 2.2 | GRPO, DanceGRPO, MixGRPO |
| Autoregressive (VLM / LLM) | Qwen-VL, Qwen3 | GRPO, SPO-DPPO |

Additional model packages (e.g. FLUX.2-Klein, HunyuanVideo 1.5) live under
`unirl/models/` without shipped recipes yet — see `unirl/models/README.md`.

## Start Here

Install the core training, inference, and evaluation dependencies:

```bash
pip install -e ".[train,infer,eval]" --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For development tools:

```bash
pip install -e ".[train,infer,eval,dev]" --no-build-isolation
```

Recipes read data, checkpoint, and W&B settings from the environment via
`${oc.env:...}`. Pass absolute paths through launcher env vars such as
`PRETRAINED_MODEL`, `DATA_PATH`, and `EVAL_DATA_PATH`, plus the W&B knobs
(`REPORT_TO_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`). Sample prompt lists are
committed under `datasets/`.

Run a single-node recipe — the first argument is a bucket-qualified recipe name
from `recipes/`:

```bash
bash scripts/run_experiment_single_node.sh diffusion_rl/sd3_trainside
```

The diffusion entrypoint is the default; select another with `ENTRY`:

```bash
ENTRY=train_vlm bash scripts/run_experiment_single_node.sh vlm_rl/argrpo_qwen_vl_geo3k_mc_4x8
ENTRY=train_pe  bash scripts/run_experiment_single_node.sh pe_rl/pe_trainside_pickscore
```

Run a multi-node recipe (taiji platform):

```bash
bash scripts/run_experiment_multinode_taiji.sh diffusion_rl/sd3_sglang_native_colocate
```

Invoke an entrypoint directly:

```bash
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_trainside num_devices=8
```

Compose and resolve a recipe without launching Ray work — checks that the
config composes and every `${oc.env:...}` resolves (it does not instantiate
`_target_`s, so it won't catch a bad target):

```bash
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_trainside --cfg job --resolve
```

## Documentation Site

The Fumadocs site in `docs/` is the main narrative documentation surface:

Published GitHub Pages build: [https://haonan3.github.io/DiffusionRL/](https://haonan3.github.io/DiffusionRL/)

```bash
cd docs
npm install
npm run dev
```

Start with `/en/docs` for English, `/zh/docs` for Chinese, and
`/en/docs/agents` for task-oriented reading order. After `npm run build`, agents
can read `/llms.txt`, `/llms-full.txt`, or focused pages under
`/md/<slug>/index.md` from the exported site.

Module READMEs remain the source of truth for code-adjacent contracts:

| Task | Read |
|---|---|
| Understand the code architecture and module boundaries | `unirl/README.md` |
| Understand Hydra config groups and where knobs belong | `unirl/config/README.md` |
| Work on rollout modes, rollout engines, or `RolloutReq` / `RolloutResp` flow | `unirl/rollout/README.md` |
| Work on the train stack — FSDP backend, mini-batch loop, LoRA/NFT inject, EMA shadow | `unirl/train/readme.md` |
| Add or debug GRPO / NFT / DPPO loss logic | `unirl/algorithms/README.md` |
| Understand SDE step kernels, σ schedule, or initial noise | `unirl/sde/README.md` |
| Add or debug reward components | `unirl/reward/README.md` |
| Add or debug a trainer→rollout weight-sync backend | `unirl/distributed/weight_sync/README.md` |
| Add or debug model code packages | `unirl/models/README.md` |

Run `npm run sync:readmes` in `docs/` to expose these module READMEs in the
Fumadocs README Reference section.

For GitHub Pages project hosting, the workflow sets
`NEXT_PUBLIC_BASE_PATH=/DiffusionRL` so Next.js generates links and assets for
the repository subpath instead of the domain root.

## Runtime Shape

Each domain has its own entrypoint, all driven the same way:

```bash
python -m unirl.train_diffusion --config-name=diffusion_rl/<recipe>      # diffusion image/video
python -m unirl.train_vlm       --config-name=vlm_rl/<recipe>            # autoregressive VLM (Qwen-VL)
python -m unirl.train_vlm       --config-name=llm_rl/<recipe>            # autoregressive LLM (Qwen3)
python -m unirl.train_pe        --config-name=pe_rl/<recipe>             # prompt-enhancer (PE)
python -m unirl.train_hi3       --config-name=unified_model_rl/<recipe>  # HunyuanImage3 (unified AR + diffusion)
```

Recipes are self-contained YAML files under `recipes/<bucket>/` selected with
`--config-name=<bucket>/<recipe>`. The five buckets are `diffusion_rl`, `vlm_rl`,
`llm_rl`, `pe_rl`, and `unified_model_rl`; each recipe carries a `# @package
_global_` header so its keys compose at the config root. At startup an entrypoint:

1. registers Hydra config dataclasses from the `unirl` package;
2. composes the chosen `recipes/<bucket>/<recipe>.yaml`;
3. builds the trainer, which constructs the typed config objects (whose
   `__post_init__` checks validate per-field invariants), acquires a Ray
   `DevicePool`, and constructs the rollout and train workers;
4. runs the rollout → reward → advantage → train → optional weight-sync loop.

Deployment modes — set by the rollout engine `_target_` and the optional `sync:`
section:

| Mode | Meaning |
|---|---|
| Direct sampling | rollout uses the `trainside` engine; the train workers also sample, so no `sync:` section is allowed |
| Separate | rollout and train workers use different GPU pools; a `sync:` variant is required |
| Colocate | rollout and train workers share GPU bundles with explicit offload/onload and weight sync |

Each `recipes/<bucket>/<recipe>.yaml` is the source of truth for model, algorithm, rollout
engine, placement, reward, sync, and batch geometry. Launchers under `scripts/`
stay thin and only prepare the runtime.

## Common Checks

```bash
# Compose one recipe and print the resolved config
python -m unirl.train_diffusion --config-name=diffusion_rl/<recipe> --cfg job --resolve

# Python syntax check
python -m compileall -q unirl

# Shell launcher syntax check
for f in scripts/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```

## Citation

```bibtex
@misc{unirl_github,
  title        = {UniRL: A Distributed RL Framework for Unified Multimodal Generative Models},
  year         = {2025},
  howpublished = {\url{https://github.com/haonan3/DiffusionRL}},
  note         = {GitHub repository},
}
```
