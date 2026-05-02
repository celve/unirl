#!/bin/bash
# =============================================================================
# Generic multi-node training launcher.
#
# Usage:
#   bash scripts/train.sh <experiment> [role] [hydra overrides...]
#
# Examples:
#   # Auto mode on TaiJi/Jizhi (each node runs the same command):
#   bash scripts/train.sh flowgrpo_fast_sd3
#   bash scripts/train.sh flowgrpo_fast_sd3 auto
#
#   # Explicit roles for manual orchestration:
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.1 bash scripts/train.sh flowgrpo_fast_sd3 head
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.2 bash scripts/train.sh flowgrpo_fast_sd3 worker
#   HEAD_IP=10.0.0.1               bash scripts/train.sh flowgrpo_fast_sd3 train \
#       run.num_rollouts=100 algorithm.kl_coef=0.04
#
#   # Cluster utilities:
#   bash scripts/train.sh _ stop
#   bash scripts/train.sh _ status
#
# Cluster platform env auto-detection (TaiJi/Jizhi):
#   CHIEF_IP     -> HEAD_IP       INDEX        -> ROLE (0=head, >0=worker)
#   LOCAL_IP     -> NODE_IP       HOST_NUM     -> NUM_NODES
#   HOST_GPU_NUM -> GPUS_PER_NODE
#
# Per-experiment wrappers (reproduce_scripts/*.sh) set environment-derived
# values (paths, wandb identity, GPU split) and exec into this launcher.
# Recipe-defining values (model, reward, algorithm, batch geometry) live in
# conf/experiment/<exp>.yaml.
#
# The launcher translates env vars to Hydra CLI overrides appended to the
# train invocation. Precedence: user "$@" CLI overrides > launcher overrides
# > YAML defaults. Empty/unset env vars emit no override (YAML default wins).
#
# Env vars translated to Hydra overrides:
#   NUM_NODES, GPUS_PER_NODE     -> placement.num_{train,rollout}_{nodes,gpus_per_node}
#                                   (only when env is user-set, not launcher fallback)
#   TRAINING_GPUS_PER_NODE,
#   ROLLOUT_GPUS_PER_NODE        -> placement.num_{train,rollout}_gpus_per_node
#                                   (asymmetric split; wins over GPUS_PER_NODE)
#   OUTPUT_DIR                   -> resume.output_dir
#   WANDB_RUN_NAME, WANDB_ENTITY -> logging.run_name, logging.entity
#   DATA_PATH, EVAL_DATA_PATH    -> run.data_path, run.eval_data_path
#
# Env vars consumed by the launcher only (NOT translated):
#   REPORT_TO_WANDB              gates check_wandb_auth (does NOT set
#                                logging.report_to_wandb — that's per-experiment YAML)
#   HEAD_IP/CHIEF_IP, NODE_IP/LOCAL_IP, INDEX, HEAD_PORT, RAY_PLACEMENT_STRATEGY
# =============================================================================

set -euo pipefail

# ── NCCL environment variables for multi-node IB/RDMA ──
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_CHECKS_DISABLE="${NCCL_CHECKS_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bond1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export SHARP_COLL_ENABLE_SAT="${SHARP_COLL_ENABLE_SAT:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Args ──
EXPERIMENT="${1:-}"
if [ -z "${EXPERIMENT}" ]; then
    echo "ERROR: experiment name is required (first positional arg)." >&2
    echo "Usage: bash scripts/train.sh <experiment> [role] [hydra overrides...]" >&2
    exit 1
fi
shift

# Role parsing: only consume the next arg if it matches a known role name.
# Anything else (e.g. a Hydra override like ``algorithm.kl_coef=0.1``) stays in
# ``"$@"`` and ROLE defaults to ``auto``. Lets users skip typing ``auto`` when
# they only want to pass overrides through.
case "${1:-}" in
    auto|auto_head|auto_worker|head|worker|train|stop|status)
        ROLE="$1"
        shift
        ;;
    *)
        ROLE="auto"
        ;;
esac

# Resolve auto → auto_head / auto_worker via INDEX (TaiJi/Jizhi).
if [ "${ROLE}" = "auto" ]; then
    if [ -n "${INDEX:-}" ]; then
        if [ "${INDEX}" = "0" ]; then ROLE="auto_head"; else ROLE="auto_worker"; fi
    fi
fi

# ── Cluster topology auto-detection ──
# Track whether topology was explicitly supplied (by wrapper or cluster platform)
# so we can emit Hydra overrides only when set — otherwise YAML/dataclass defaults
# win, preserving recipe-specific splits like flowgrpo_sglang_sd3_separate's 4/4.
NUM_NODES_USER_SET=0; [ -n "${NUM_NODES:-}${HOST_NUM:-}" ] && NUM_NODES_USER_SET=1
GPUS_PER_NODE_USER_SET=0; [ -n "${GPUS_PER_NODE:-}${HOST_GPU_NUM:-}" ] && GPUS_PER_NODE_USER_SET=1

NUM_NODES="${NUM_NODES:-${HOST_NUM:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
HEAD_IP="${HEAD_IP:-${CHIEF_IP:-}}"
NODE_IP="${NODE_IP:-${LOCAL_IP:-}}"
HEAD_PORT="${HEAD_PORT:-6379}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
RAY_PLACEMENT_STRATEGY="${RAY_PLACEMENT_STRATEGY:-SPREAD}"
WORKER_WAIT_INTERVAL="${WORKER_WAIT_INTERVAL:-10}"
WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-600}"

# ── Output / wandb defaults (env-overridable; wrappers usually set these) ──
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXPERIMENT}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}-${NUM_NODES}node}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"

# ── Helpers ──
print_usage() {
    cat <<EOF
Usage:
  $(basename "$0") <experiment> [role] [hydra overrides...]

Roles:
  auto              auto-detect head/worker via INDEX (default)
  head              start ray head and exit
  worker            join ray cluster as worker (sleeps after join)
  train             submit training to existing ray cluster (no ray start)
  stop              ray stop on local node
  status            ray status against the cluster

Cluster env auto-detection (TaiJi/Jizhi):
  CHIEF_IP     -> HEAD_IP       INDEX        -> ROLE (0=head, >0=worker)
  LOCAL_IP     -> NODE_IP       HOST_NUM     -> NUM_NODES
  HOST_GPU_NUM -> GPUS_PER_NODE
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then echo "ERROR: command not found: $1" >&2; exit 1; fi
}

wait_for_workers() {
    local expected="$1" elapsed=0
    echo "Waiting for ${expected} nodes to join the Ray cluster..."
    while true; do
        local n; n=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "0")
        if [ "${n}" -ge "${expected}" ]; then echo "All ${expected} nodes connected."; return 0; fi
        if [ "${elapsed}" -ge "${WORKER_WAIT_TIMEOUT}" ]; then
            echo "WARNING: Timeout (${WORKER_WAIT_TIMEOUT}s). Got ${n}/${expected} nodes. Proceeding anyway."; return 0
        fi
        echo "  ${n}/${expected} nodes ready, waiting... (${elapsed}s/${WORKER_WAIT_TIMEOUT}s)"
        sleep "${WORKER_WAIT_INTERVAL}"; elapsed=$(( elapsed + WORKER_WAIT_INTERVAL ))
    done
}

run_training() {
    source "${SCRIPT_DIR}/_check_wandb.sh"
    check_wandb_auth "${REPORT_TO_WANDB:-false}"

    mkdir -p "${OUTPUT_DIR:-/tmp/diffusionrl/output}"
    echo "Submitting ${EXPERIMENT} to Ray cluster ${HEAD_IP}:${HEAD_PORT}"
    echo "Topology: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs"
    echo "Output dir: ${OUTPUT_DIR:-(experiment default)}"

    export RAY_ADDRESS="${HEAD_IP}:${HEAD_PORT}"

    # Build Hydra CLI overrides from env-derived values. Wrappers under
    # reproduce_scripts/*.sh set these env vars; we translate to overrides so
    # YAML configs hold no ${oc.env:...} interpolations.
    #
    # Order: GPUS_PER_NODE-derived first; TRAINING_GPUS_PER_NODE / ROLLOUT_GPUS_PER_NODE
    # after (asymmetric split wins). User "$@" appended last beats both.
    local -a hydra_overrides=()

    if [ "${NUM_NODES_USER_SET}" = "1" ]; then
        hydra_overrides+=("placement.num_train_nodes=${NUM_NODES}" "placement.num_rollout_nodes=${NUM_NODES}")
    fi
    if [ "${GPUS_PER_NODE_USER_SET}" = "1" ]; then
        hydra_overrides+=("placement.num_train_gpus_per_node=${GPUS_PER_NODE}" "placement.num_rollout_gpus_per_node=${GPUS_PER_NODE}")
    fi
    [ -n "${TRAINING_GPUS_PER_NODE:-}" ] && hydra_overrides+=("placement.num_train_gpus_per_node=${TRAINING_GPUS_PER_NODE}")
    [ -n "${ROLLOUT_GPUS_PER_NODE:-}" ]  && hydra_overrides+=("placement.num_rollout_gpus_per_node=${ROLLOUT_GPUS_PER_NODE}")

    [ -n "${OUTPUT_DIR:-}" ]      && hydra_overrides+=("resume.output_dir=${OUTPUT_DIR}")
    [ -n "${WANDB_RUN_NAME:-}" ]  && hydra_overrides+=("logging.run_name=${WANDB_RUN_NAME}")
    [ -n "${WANDB_ENTITY:-}" ]    && hydra_overrides+=("logging.entity=${WANDB_ENTITY}")
    [ -n "${DATA_PATH:-}" ]       && hydra_overrides+=("run.data_path=${DATA_PATH}")
    [ -n "${EVAL_DATA_PATH:-}" ]  && hydra_overrides+=("run.eval_data_path=${EVAL_DATA_PATH}")

    python -m diffusionrl.train +experiment="${EXPERIMENT}" "${hydra_overrides[@]}" "$@"
}

# ── Main ──
case "${ROLE}" in
    stop)   ray stop >/dev/null 2>&1 || true; echo "Ray stopped on local node."; exit 0 ;;
    status) if [ -n "${HEAD_IP}" ]; then ray status --address "${HEAD_IP}:${HEAD_PORT}"; else ray status; fi; exit 0 ;;
esac

if [ -z "${ROLE}" ]; then print_usage; exit 1; fi
require_cmd ray

case "${ROLE}" in
    auto_head)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP (or CHIEF_IP) is required" >&2; exit 1; fi
        : "${NODE_IP:=${HEAD_IP}}"
        ray stop >/dev/null 2>&1 || true
        echo "[INDEX=${INDEX:-0}] Starting Ray head on ${NODE_IP} (port=${HEAD_PORT}, gpus=${GPUS_PER_NODE})"
        ray start --head --node-ip-address "${NODE_IP}" --port "${HEAD_PORT}" \
            --dashboard-host "${DASHBOARD_HOST}" --num-gpus "${GPUS_PER_NODE}"
        wait_for_workers "${NUM_NODES}"
        run_training "$@"
        ;;

    auto_worker)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP (or CHIEF_IP) is required" >&2; exit 1; fi
        if [ -z "${NODE_IP}" ]; then echo "ERROR: NODE_IP (or LOCAL_IP) is required for worker" >&2; exit 1; fi
        ray stop >/dev/null 2>&1 || true
        echo "[INDEX=${INDEX:-?}] Joining Ray cluster ${HEAD_IP}:${HEAD_PORT} from ${NODE_IP} (gpus=${GPUS_PER_NODE})"
        ray start --address "${HEAD_IP}:${HEAD_PORT}" --node-ip-address "${NODE_IP}" --num-gpus "${GPUS_PER_NODE}"
        echo "Worker joined. Keeping process alive..."
        tail -f /dev/null
        ;;

    head)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP is required for role=head" >&2; exit 1; fi
        : "${NODE_IP:=${HEAD_IP}}"
        ray stop >/dev/null 2>&1 || true
        echo "Starting Ray head on ${NODE_IP} (port=${HEAD_PORT}, gpus=${GPUS_PER_NODE})"
        ray start --head --node-ip-address "${NODE_IP}" --port "${HEAD_PORT}" \
            --dashboard-host "${DASHBOARD_HOST}" --num-gpus "${GPUS_PER_NODE}"
        ;;

    worker)
        if [ -z "${HEAD_IP}" ] || [ -z "${NODE_IP}" ]; then
            echo "ERROR: HEAD_IP and NODE_IP are required for role=worker" >&2; exit 1
        fi
        ray stop >/dev/null 2>&1 || true
        echo "Joining Ray cluster ${HEAD_IP}:${HEAD_PORT} from ${NODE_IP} (gpus=${GPUS_PER_NODE})"
        ray start --address "${HEAD_IP}:${HEAD_PORT}" --node-ip-address "${NODE_IP}" --num-gpus "${GPUS_PER_NODE}"
        ;;

    train)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP is required for role=train" >&2; exit 1; fi
        run_training "$@"
        ;;

    *)
        echo "ERROR: unknown role: ${ROLE}" >&2
        print_usage
        exit 1
        ;;
esac
