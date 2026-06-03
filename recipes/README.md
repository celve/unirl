# Recipes

Self-contained Hydra recipes — one YAML per experiment. A recipe is the single
source of truth for a run: model, algorithm, rollout engine, placement, reward,
weight sync, and batch geometry, each instantiated directly by `_target_` (no
Hydra config-group overrides).

Recipes are **bucketed by trainer domain** — one subdirectory per entrypoint.
Select one with `--config-name=<bucket>/<recipe>` (the bucket directory is part
of the name; drop the `.yaml`).

> This directory replaces the old flat top-level `conf/` tree. The docs site
> ([Experiment Recipes](../docs/content/docs/en/configuration/experiments.mdx))
> is the rendered, always-current index of maintained recipes.

## Layout

| Bucket | Entrypoint | Holds |
|---|---|---|
| [`diffusion_rl/`](diffusion_rl/) | `python -m unirl.train_diffusion` | Diffusion image/video RL — SD3, Qwen-Image, Flux.2-Klein, WAN 2.1/2.2, HunyuanVideo |
| [`vlm_rl/`](vlm_rl/) | `python -m unirl.train_vlm` | Vision-language AR RL — Qwen-VL ARGRPO |
| [`llm_rl/`](llm_rl/) | `python -m unirl.train_vlm` | Text-only AR RL — Qwen3 DRPO (shares the AR entrypoint with `vlm_rl/`) |
| [`pe_rl/`](pe_rl/) | `python -m unirl.train_pe` | Prompt-enhancer (AR + diffusion) |
| [`unified_model_rl/`](unified_model_rl/) | `python -m unirl.train_unified_model` | Unified AR + diffusion — HunyuanImage3 |

## Launching

```bash
# direct entrypoint (compose-check first with `--cfg job --resolve`)
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_trainside --cfg job --resolve
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_trainside

# via the launchers — arg 1 is the bucketed recipe name; ENTRY picks a non-diffusion entrypoint
bash scripts/run_experiment_single_node.sh diffusion_rl/sd3_trainside
ENTRY=train_vlm bash scripts/run_experiment_single_node.sh vlm_rl/argrpo_qwen_vl_geo3k_mc_4x8
bash scripts/run_experiment_multinode_taiji.sh diffusion_rl/sd3_sglang_native_colocate
```

Every recipe **must start with `# @package _global_`** on line 1. Recipes live in
a subdirectory, so without it Hydra would nest the whole config under the bucket
key (e.g. `diffusion_rl.num_devices`) and the entrypoint's top-level fields would
be missing. Cluster-local paths, model mounts, output dirs, and W&B identity stay
out of the YAML — pass them as env vars / CLI overrides; recipes read them with
`${oc.env:...}`.

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

Bucket-specific trailing qualifiers extend the chain:

- **`pe_rl/`** appends the reward: `pe_sglang_full_pickscore`, `pe_sglang_full_wise`.
- **`vlm_rl/`** appends dataset + task: `argrpo_qwen_vl_geo3k_mc_4x8` (`geo3k` · multiple-choice).
- AR buckets append the cluster shape `<N>x<G>` (nodes × GPUs): `..._4x8`.

**Known deviation:** the AR recipes in `vlm_rl/`/`llm_rl/` predate this schema and
lead with the algorithm (`argrpo_qwen_vl_…`, `ar_drpo_qwen3_…`) rather than the
model. They are kept as-is; new recipes should follow the model-first order above.

## Adding a recipe

1. Copy the closest existing recipe in the right bucket.
2. Keep line 1 as `# @package _global_`; name the file per the schema above.
3. Keep every choice in YAML, instantiated by `_target_`; use `${oc.env:...}` only
   for deployment-specific paths and logging identity.
4. Compose-check: `python -m unirl.train_<entry> --config-name=<bucket>/<recipe> --cfg job --resolve`.
5. List it on the [Experiment Recipes](../docs/content/docs/en/configuration/experiments.mdx) page.
