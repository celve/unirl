# Examples

Self-contained Hydra recipes — one YAML per experiment. A recipe is the single
source of truth for a run: model, algorithm, rollout engine, placement, reward,
weight sync, and batch geometry, each instantiated directly by `_target_` (no
Hydra config-group overrides).

Recipes are grouped by trainer domain, then model. Select one with
`--config-name=<domain>/<model>/<recipe>` (drop the `.yaml`).

> This directory replaces the old top-level `recipes/` tree.

## Layout

| Domain | Model dirs | Entrypoint | Holds |
|---|---|---|---|
| [`diffusion/`](diffusion/) | `sd3/`, `qwen_image/`, `flux2_klein/`, `wan21/`, `wan22/`, `hunyuan_video/`, `hunyuan_video15/` | `python -m unirl.train_diffusion` | Diffusion image/video RL — SD3, Qwen-Image, Flux.2-Klein, WAN 2.1/2.2, HunyuanVideo |
| [`vlm/`](vlm/) | `qwen_vl/` | `python -m unirl.train_vlm` | Vision-language AR RL — Qwen-VL ARGRPO |
| [`llm/`](llm/) | `qwen3/` | `python -m unirl.train_vlm` | Text-only AR RL — Qwen3 DRPO (shares the AR entrypoint with `vlm/`) |
| [`pe/`](pe/) | `pe/` | `python -m unirl.train_pe` | Prompt-enhancer (AR + diffusion) |
| [`unified_model/`](unified_model/) | `hi3/` | `python -m unirl.train_unified_model` | Unified AR + diffusion — HunyuanImage3 |

## Launching

The bash launchers live in this directory. The first argument is the
domain/model-qualified recipe name (passed to Hydra as `--config-name`); any extra
args are forwarded verbatim as Hydra overrides. `ENTRY` selects a non-diffusion
entrypoint (`train_vlm` / `train_pe` / `train_unified_model`); the default is
`train_diffusion`.

```bash
# 0. Compose-check first — verifies the config composes and every ${oc.env:...} resolves
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve

# 1. Single node
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
ENTRY=train_vlm bash examples/run_experiment_single_node.sh vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8
ENTRY=train_pe  bash examples/run_experiment_single_node.sh pe/pe/pe_trainside_pickscore

# 2. Multi-node (taiji)
bash examples/run_experiment_multinode_taiji.sh diffusion/sd3/sd3_sglang_native_colocate

# 3. Or invoke an entrypoint directly, without the launchers
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8
```

Pass cluster-local paths and W&B identity through env vars (`PRETRAINED_MODEL`,
`DATA_PATH`, `EVAL_DATA_PATH`, `REPORT_TO_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`).
The mooncake-backed recipe (`*_tq_mooncake`) needs its metadata server up first —
start it on the head node with `bash examples/mooncake_master.sh start` before launching.

Every recipe **must start with `# @package _global_`** on line 1. Recipes live in
subdirectories, so without it Hydra would nest the whole config under the domain
and model keys (e.g. `diffusion.sd3.num_devices`) and the entrypoint's top-level
fields would be missing. Cluster-local paths, model mounts, output dirs, and W&B
identity stay out of the YAML — pass them as env vars / CLI overrides; recipes
read them with `${oc.env:...}`.

## Naming schema

A recipe filename is a fixed-order, `_`-joined chain of segments. Every segment
except `model` is optional and is **omitted when it is the default or does not
apply** — so a name carries only what distinguishes it from its siblings, and
related recipes sort together.

```
<model>[_<task>][_<size>][_<algorithm>][_<engine>][_<adapter>][_<topology>]
```

| Segment | Position | Values (examples) | Omit when |
|---|---|---|---|
| `model` | required, first | `sd3`, `qwen_image`, `flux2_klein`, `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video15`, `hi3` | never |
| `task` | after model | `t2v`, `i2v` | text-to-image (the implicit default) |
| `size` | after task | `14b` | only one size in the family |
| `algorithm` | middle | `dancegrpo`, `mixgrpo`, `nft`, `flowdppo` | plain GRPO (the default) |
| `engine` | after algorithm | `trainside`, `sglang`, `vllmomni` | — |
| `adapter` | after engine | `full`, `lora` | unambiguous from the rest |
| `topology` | last | placement `colocate`/`separate`; sync `nccl`/`tensor`/`ipc`; engine mode `native`/`replay` | single-slab colocate default |

Worked examples:

| Recipe | Reads as |
|---|---|
| `sd3_trainside` | SD3 · trainside engine · (default GRPO) |
| `sd3_nft_sglang` | SD3 · NFT · SGLang engine |
| `qwen_image_dancegrpo` | Qwen-Image · DanceGRPO |
| `wan22_t2v_14b_dancegrpo` | WAN 2.2 · text-to-video · 14B · DanceGRPO |
| `hunyuan_video15_t2v_dancegrpo_trainside` | HunyuanVideo-1.5 · text-to-video · DanceGRPO · trainside engine |
| `sd3_vllmomni_full_nccl_separate` | SD3 · vLLM-Omni engine · full-weight · NCCL sync · separate slabs |

Domain-specific trailing qualifiers extend the chain:

- **`pe/`** appends the reward: `pe_sglang_full_pickscore`, `pe_sglang_full_wise`.
- **`vlm/`** appends dataset + task: `argrpo_qwen_vl_geo3k_mc_4x8` (`geo3k` · multiple-choice).
- AR domains append the cluster shape `<N>x<G>` (nodes × GPUs): `..._4x8`.

**Known deviation:** the AR recipes in `vlm/`/`llm/` predate this schema and
lead with the algorithm (`argrpo_qwen_vl_…`, `ar_drpo_qwen3_…`) rather than the
model. They are kept as-is; new recipes should follow the model-first order above.

## Adding a recipe

1. Copy the closest existing recipe in the right domain / model directory.
2. Keep line 1 as `# @package _global_`; name the file per the schema above.
3. Keep every choice in YAML, instantiated by `_target_`; use `${oc.env:...}` only
   for deployment-specific paths and logging identity.
4. Compose-check: `python -m unirl.train_<entry> --config-name=<domain>/<model>/<recipe> --cfg job --resolve`.

## Development checks

Before opening a PR, run the checks that match the files you touched:

```bash
# Compose one changed or representative recipe and print the resolved config
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve

# Python syntax check
python -m compileall -q unirl

# Shell launcher syntax check
for f in examples/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```
