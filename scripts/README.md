# Training Scripts

`scripts/` is intentionally thin. Training semantics live in
`conf/experiment/*.yaml`; shell scripts only prepare the runtime, start Ray,
set path/logging defaults, and forward Hydra overrides.

For the experiment inventory and config ownership rules, see
`../conf/README.md`.

## Launchers

| Script | Use |
|---|---|
| `run_experiment_single_node.sh <experiment>` | Cluster-agnostic 1-node launcher for any `conf/experiment/<experiment>.yaml` (no platform env needed) |
| `run_experiment_multinode_taiji.sh <experiment>` | Multi-node launcher for the **taiji** platform; the head runs the training driver, other nodes join Ray. Brought up via SPMD (default) or head-side ssh fan-out — see [Multi-node launch](#multi-node-launch-launch) |

Examples:

```bash
bash scripts/run_experiment_single_node.sh grpo_wan21_t2v
bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate
bash scripts/run_experiment_multinode_taiji.sh flowgrpo_fast_qwen_image_2x8
bash scripts/run_experiment_multinode_taiji.sh nft_qwen_image_4x8 run.num_rollouts=100
DRY_RUN=1 bash scripts/run_experiment_single_node.sh mixgrpo_sd35
```

The generic launchers set these environment-derived overrides:

- `run.data_path=${DATA_PATH}`
- `run.eval_data_path=${EVAL_DATA_PATH}`
- `resume.output_dir=${OUTPUT_DIR}`
- `logging.run_name=${WANDB_RUN_NAME}`
- `logging.report_to_wandb=${REPORT_TO_WANDB}`
- `logging.entity=${WANDB_ENTITY}` when `WANDB_ENTITY` is set

Model checkpoint env vars remain recipe-specific, for example
`PRETRAINED_MODEL`, `SD3_PATH`, `QWEN_IMAGE_PATH`, and
`HUNYUAN_VIDEO15_PATH`.

## Data plane

Both launchers pick how each rollout's **heavy payload** (latents, log-probs,
decoded images — GB-scale) reaches the trainer via `DATA_PLANE`; the **light
metadata** (rewards, advantages — KB-scale) always goes to the **driver** (the one
process that orchestrates actors and logs). *Rollout actors* generate samples and
*train actors* run the gradient updates — in trainside / direct sampling they are the
same actors.

```
  Legend    ═══►  heavy payload   (GB · latents, log-probs, decoded images)
            ───►  light metadata  (KB · rewards, advantages, ids)
  ─────────────────────────────────────────────────────────────────────────────────────────

  ray          [ rollout actor ] ═══════►[ DRIVER · torch.cat whole batch ]═══════►[ train actor ]
  (default)                                       ⚠ one process holds it all  →  OOM at scale

  tq_simple    [ rollout actor ] ═══════►[ TransferQueue · Ray object store ]═══════►[ train actor ]
                       └───light───►[ DRIVER ]                 heavy skips the driver  ·  TCP hop

  tq_mooncake  [ rollout actor ] ═══════►[ TransferQueue · mooncake RDMA ⚡ ]═══════►[ train actor ]
                       └───light───►[ DRIVER ]                 heavy skips the driver  ·  RDMA hop

  keep_local   [ rollout actor ═══════► trains in place · same GPU ]       producer == consumer
                       └───light───►[ DRIVER ]                             nothing moves
```

| `DATA_PLANE` | When to use | Extra infra | Single node |
|---|---|---|---|
| `ray` (default) | small batches / quick start | none | ✅ |
| `tq_simple` | take the batch off the driver, at any scale | none | ✅ |
| `tq_mooncake` | production multi-node (fast cross-node) | mooncake services (the taiji launcher's head auto-starts them + derives `MOONCAKE_*` from `CHIEF_IP`); set `PROTOCOL` (rdma/tcp) | ❌ multi-node only |
| `keep_local` | trainside / direct sampling — skip transport | none | ✅ |

**Single node** (`run_experiment_single_node.sh`) needs **no taiji / platform env** and
offers **three** planes — `ray`, `tq_simple`, `keep_local`. `tq_mooncake` is **multi-node
only** (its whole win is RDMA *across nodes*, and it needs external services), so it lives
only on the taiji multi-node launcher.

**Jargon:**
- **driver** — the one orchestrator process (launches actors, dispatches, logs); not a GPU worker.
- **heavy payload / light metadata** — GB-scale tensors vs KB-scale scalars (rewards/advantages/ids).
- **gather-OOM** — in `ray` the driver `torch.cat`s the whole batch into one process → OOM at scale; the reason the other planes exist.
- **TransferQueue (TQ)** — a put/get store that passes the heavy payload actor→actor, bypassing the driver.
- **RDMA / RoCE** — the NIC moves data between machines' memory with no CPU copy (RoCE = RDMA over Ethernet).
- **host-staged** — payload copied GPU→host (`.cpu()`) before transport.
- **GPUDirect RDMA / zero-copy** — RDMA straight between GPU memories, no host copy: the ideal, needs NIC/driver support. `tq_mooncake` here is **host-staged RDMA** — zero-copy on the cross-node wire (`zero_copy.enable`) but with a GPU↔host copy at each end, *not* GPUDirect.

```bash
# single node (no taiji) — ray / tq_simple / keep_local:
DATA_PLANE=tq_simple bash scripts/run_experiment_single_node.sh <experiment>

# multi-node (taiji) — tq_mooncake auto-starts its services on the head:
DATA_PLANE=tq_mooncake PROTOCOL=rdma \
  bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
```

## Multi-node launch (`LAUNCH`)

`run_experiment_multinode_taiji.sh` brings Ray up across nodes one of two ways;
both end with the head running the single driver and every node joined to Ray.
The data plane (above) is orthogonal — all four work under either mode.

| `LAUNCH` | How nodes start | When |
|---|---|---|
| `spmd` (default) | The platform runs this same script on every node; rank 0 (`INDEX=0`) is head + driver, others join Ray and idle. Submit once; taiji fans it out. | Batch jobs |
| `ssh` | Run once on the head; it starts the head, `ssh`'s `ray start` onto every other node in `NODE_IP_LIST`, then runs the driver. | Interactive sessions where you only have a shell on the head |

`LAUNCH=ssh` needs passwordless ssh head→workers (taiji provides it), the repo at
the **same path** on every node (shared mount), and `CONDA_ENV` set so a non-login
ssh shell finds `ray`. Per-worker join logs land in `/tmp/diffusionrl_ray_worker_*.log`.

```bash
LAUNCH=ssh DATA_PLANE=keep_local \
  bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
```

## What Goes Where

| Layer | What belongs there |
|---|---|
| `conf/experiment/<exp>.yaml` | Model, algorithm, reward, rollout engine, sync, placement, batch geometry, LoRA/FSDP/EMA/NFT policy choices |
| YAML env interpolation | Checkpoint/data/output paths and logging identity when those are deployment-specific |
| `scripts/*.sh` | Python env activation, Ray startup, per-job path defaults, and forwarding CLI overrides |

Override precedence is:

```text
CLI Hydra override > launcher env var > YAML default
```

For different cluster geometry, override both placement and training batch
geometry together:

```bash
bash scripts/run_experiment_multinode_taiji.sh flowgrpo_fast_qwen_image_4x8 \
    placement.num_train_nodes=2 \
    placement.num_rollout_nodes=2 \
    training.topology.actor_count=16 \
    training.plan.local_batch_size=36 \
    training.plan.local_mini_batch_size=18
```

`validate_training_batch_geometry` reports inconsistent geometry before Ray
work starts.

## Compose Check

Use a real training recipe for config validation:

```bash
python -m diffusionrl.train +experiment=<experiment> --cfg job --resolve
```
