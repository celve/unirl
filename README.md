# DiffusionRL

DiffusionRL is a distributed reinforcement learning framework for diffusion
model optimization. It trains diffusion and multimodal generation models with
Ray actor groups, Hydra experiment recipes, composable training policies, and
pluggable rollout engines.

```text
   prompts ──▶ rollout actor group ──▶ rewards ──▶ advantages ──┐
                  (samples x_T → x_0)                            │
                                                                 ▼
                                                          train actor group
                                                          (policy stack +
                                                           stage algorithms)
                                                                 │
                                          weight sync ◀──────────┘
                                          (dedicated rollout only)
```

| Domain | Models | Recipes |
|---|---|---|
| Image | Stable Diffusion 3, Qwen-Image, HunyuanImage3 | GRPO, FlowGRPO, DanceGRPO, MixGRPO, NFT |
| Video | WAN 2.1 / 2.2, HunyuanVideo 1.5 | GRPO |

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

Toy prompt files are committed under `data/samples/`. For real data or model
checkpoints, pass absolute paths through launcher env vars such as `DATA_PATH`,
`EVAL_DATA_PATH`, `OUTPUT_DIR`, and `PRETRAINED_MODEL`.

Run a single-node recipe:

```bash
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate
```

Run a multi-node recipe (taiji platform):

```bash
bash scripts/run_experiment_multinode_taiji.sh flowgrpo_fast_qwen_image_2x8
```

Invoke the Hydra entrypoint directly:

```bash
python -m diffusionrl.train \
    +experiment=flowgrpo_fast_sd3_colocate \
    run.data_path=data/samples/prompts_toy.json \
    resume.output_dir=outputs/my_experiment
```

Validate a recipe without launching Ray work:

```bash
python -m diffusionrl.train +experiment=flowgrpo_fast_sd3_colocate --cfg job --resolve
```

## Documentation Map

Read these files by task:

| Task | Read |
|---|---|
| Pick and launch a training recipe | `scripts/README.md` |
| Understand Hydra config groups and where knobs belong | `conf/README.md` |
| Understand the code architecture and module boundaries | `diffusionrl/README.md` |
| Work on Ray actors, actor groups, placement, or colocate orchestration | `diffusionrl/ray/README.md` |
| Work on rollout modes, rollout engines, or `RolloutReq` / `RolloutResp` flow | `diffusionrl/rollout/README.md` |
| Work on train actors, policy composition, FSDP, LoRA, EMA, or NFT policies | `diffusionrl/training/README.md` |
| Add or debug GRPO / NFT-style loss logic | `diffusionrl/algorithms/README.md` |
| Understand SDE step kernels, σ schedule, or initial noise | `diffusionrl/sde/README.md` |
| Add or debug reward components | `diffusionrl/reward/README.md` |
| Add or debug a trainer→rollout weight-sync backend | `diffusionrl/distributed/weight_sync/README.md` |
| Add or debug model code packages | `diffusionrl/models/README.md` |
| Add or mount model artifacts | `models/README.md` |
| Add or mount datasets | `data/README.md` |
| Run HI3 / vLLM-Omni recipes that need external patches | `patches/README.md` |

This layout is intentional: the root README gives the shortest path to running
the project, while module READMEs describe the contracts closest to the code
that owns them.

## Runtime Shape

The maintained entrypoint is:

```bash
python -m diffusionrl.train +experiment=<name>
```

At startup it:

1. registers Hydra config dataclasses from the `diffusionrl` package;
2. composes `conf/train.yaml` plus `conf/experiment/<name>.yaml`;
3. validates cross-component contracts before Ray actors are created;
4. creates placement, rollout actor group, and train actor group;
5. runs rollout, reward, advantage, train, and optional weight-sync phases.

Deployment modes:

| Mode | Meaning |
|---|---|
| Direct sampling | `rollout/engine: trainside`; train actors also sample, so no `sync:` section is allowed |
| Separate | rollout actors and train actors use different GPU pools; a `sync:` variant is required |
| Colocate | rollout and train actors share GPU bundles with explicit offload/onload and weight sync |

Experiment YAMLs under `conf/experiment/` are the source of truth for model,
algorithm, rollout engine, placement, reward, sync, and batch geometry.
Launchers under `scripts/` should stay thin and only prepare the runtime.

## Common Checks

```bash
# Compose one recipe and print the resolved config
python -m diffusionrl.train +experiment=<experiment> --cfg job --resolve

# Python syntax check
python -m compileall -q diffusionrl

# Shell launcher syntax check
for f in scripts/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```

## Citation

```bibtex
@misc{diffusionrl_github,
  title        = {DiffusionRL: A Distributed RL Framework for Diffusion Model Optimization},
  year         = {2025},
  howpublished = {\url{https://github.com/your-org/diffusionrl}},
  note         = {GitHub repository},
}
```
