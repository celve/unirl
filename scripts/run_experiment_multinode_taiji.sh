#!/usr/bin/env bash
#
# Multi-node experiment launcher for the taiji platform. The platform runs this
# SAME script on every node (SPMD): rank 0 starts the Ray head and the training
# driver; every other rank joins Ray and idles. Cluster topology defaults to
# taiji's job env (see "Cluster topology" below); set the explicit vars to run
# on any other cluster.
#
# Data plane (how rollout samples reach the trainer) is selected with DATA_PLANE:
#   ray         (default) driver gathers rollouts over the Ray object store
#   tq_simple   TransferQueue on Ray-backed host storage (off-driver, zero infra)
#   tq_mooncake TransferQueue over mooncake RDMA (production) — needs an EXTERNAL
#               mooncake_master + http_metadata_server on the head; set MOONCAKE_*
#   keep_local  direct-sampling actors keep rollouts local; only light metadata
#               crosses to the driver (no transfer at all)
#
# Submit once as the platform's multi-node job entrypoint (it fans out to every
# node and sets INDEX + CHIEF_IP). Examples (same line on every node):
#   bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
#   DATA_PLANE=tq_simple bash scripts/run_experiment_multinode_taiji.sh <experiment>
#   DATA_PLANE=tq_mooncake PROTOCOL=rdma \
#     MOONCAKE_METADATA_URL=http://$CHIEF_IP:8080/metadata \
#     MOONCAKE_MASTER_ADDR=$CHIEF_IP:50051 \
#     bash scripts/run_experiment_multinode_taiji.sh grpo_flux2_klein9b_trainside_2x8
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${EXPERIMENT:-}" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Usage: $0 <experiment> [hydra overrides...]"
        exit 2
    fi
    EXPERIMENT="$1"
    shift
fi

# --- Python env (optional) --------------------------------------------------
if [ -n "${CONDA_ENV:-}" ]; then
    if [ -f "${CONDA_SH:-}" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
    elif [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/data/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}"
elif [ -n "${VENV_DIR:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# --- Path / logging defaults ------------------------------------------------
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXPERIMENT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

CMD=(
    python -m diffusionrl.train
    "+experiment=${EXPERIMENT}"
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
    "logging.run_name=${WANDB_RUN_NAME}"
    "logging.report_to_wandb=${REPORT_TO_WANDB}"
)
if [ -n "${WANDB_ENTITY:-}" ]; then
    CMD+=("logging.entity=${WANDB_ENTITY}")
fi

# --- Data-plane selection (DATA_PLANE) --------------------------------------
# Append the Hydra overrides that pick how rollout data reaches the trainer.
DATA_PLANE="${DATA_PLANE:-ray}"
case "${DATA_PLANE}" in
    ray)
        : # default driver gather — no override
        ;;
    tq_simple)
        CMD+=("+transfer_queue=simple")
        ;;
    tq_mooncake)
        if [ -z "${MOONCAKE_METADATA_URL:-}" ] || [ -z "${MOONCAKE_MASTER_ADDR:-}" ]; then
            echo "DATA_PLANE=tq_mooncake needs MOONCAKE_METADATA_URL=http://<HEAD_IP>:<port>/metadata" >&2
            echo "and MOONCAKE_MASTER_ADDR=<HEAD_IP>:<rpc_port> (external mooncake_master on the head)." >&2
            exit 2
        fi
        CMD+=(
            "+transfer_queue=mooncake_tuned"
            "transfer_queue.protocol=${PROTOCOL:-rdma}"
            "transfer_queue.metadata_server=${MOONCAKE_METADATA_URL}"
            "transfer_queue.master_server_address=${MOONCAKE_MASTER_ADDR}"
        )
        ;;
    keep_local)
        CMD+=("training.execution.keep_local=true")
        ;;
    *)
        echo "Unknown DATA_PLANE='${DATA_PLANE}' (use ray|tq_simple|tq_mooncake|keep_local)" >&2
        exit 2
        ;;
esac

CMD+=("$@")

echo "Command (DATA_PLANE=${DATA_PLANE}):"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

# --- Cluster topology (taiji platform defaults) -----------------------------
# Defaults come from taiji's multi-node job env (commented per line); the
# explicit vars always win, so this launcher also runs on a non-taiji cluster.
NUM_NODES="${NUM_NODES:-${HOST_NUM:-2}}"               # taiji HOST_NUM:     node count
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"   # taiji HOST_GPU_NUM: GPUs per node
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"                  # taiji INDEX:        this node's rank
RAY_PORT="${RAY_PORT:-6379}"

# This node's IP. Prefer an explicit NODE_IP, else taiji's LOCAL_IP. On multi-NIC
# / container nodes `hostname -I` often returns a container-internal IP that peers
# can't reach, so when CHIEF_IP is known, pick this node's IP on the chief's /16.
all_ips="$(hostname -I 2>/dev/null || true)"
if [ -z "${NODE_IP:-}" ] && [ -n "${LOCAL_IP:-}" ]; then
    NODE_IP="${LOCAL_IP}"
fi
if [ -z "${NODE_IP:-}" ] && [ -n "${CHIEF_IP:-}" ]; then
    chief_subnet="$(echo "${CHIEF_IP}" | cut -d. -f1-2)"
    NODE_IP="$(echo "${all_ips}" | tr ' ' '\n' | grep "^${chief_subnet}\." | head -1 || true)"
fi
if [ -z "${NODE_IP:-}" ]; then
    NODE_IP="$(echo "${all_ips}" | awk '{print $1}')"
fi
NODE_IP="${NODE_IP:-127.0.0.1}"

HEAD_IP="${HEAD_IP:-${CHIEF_IP:-${NODE_IP}}}"          # taiji CHIEF_IP:     head node IP

mkdir -p "${OUTPUT_DIR}"
ray stop >/dev/null 2>&1 || true

if [ "${NODE_RANK}" = "0" ]; then
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --num-gpus="${GPUS_PER_NODE}"
else
    until ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${GPUS_PER_NODE}"; do
        echo "Ray head not ready yet; retrying in 5s..."
        sleep 5
    done
    echo "Worker node joined Ray cluster; head node owns the training driver."
    tail -f /dev/null
fi

echo "Ray cluster target: ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s)"
sleep "${RAY_CLUSTER_WAIT_S:-30}"
exec "${CMD[@]}"
