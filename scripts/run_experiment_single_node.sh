#!/usr/bin/env bash
#
# Single-node experiment launcher (cluster-agnostic — no platform env needed).
# Starts a local Ray head on this machine and runs the training driver.
#
# Data plane (how rollout samples reach the trainer) is selected with DATA_PLANE:
#   ray         (default) driver gathers rollouts over the Ray object store
#   tq_simple   TransferQueue on Ray-backed host storage (off-driver, zero infra)
#   keep_local  direct-sampling actors keep rollouts local; only light metadata
#               crosses to the driver (no transfer at all)
#
# tq_mooncake (RDMA) is intentionally NOT offered here: it only pays off across
# nodes and needs external mooncake services. On a single node use tq_simple;
# for the mooncake data plane use scripts/run_experiment_multinode_taiji.sh.
#
# Example:
#   bash scripts/run_experiment_single_node.sh flowgrpo_fast_sd3_colocate
#   DATA_PLANE=tq_simple bash scripts/run_experiment_single_node.sh grpo_wan21_t2v
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
    keep_local)
        CMD+=("training.execution.keep_local=true")
        ;;
    *)
        echo "Unknown DATA_PLANE='${DATA_PLANE}' (single node: ray|tq_simple|keep_local;" >&2
        echo "tq_mooncake is multi-node only — see run_experiment_multinode_taiji.sh)." >&2
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

# --- Single-node Ray (local head) -------------------------------------------
# GPU count: override with GPUS_PER_NODE, else autodetect, else assume 8.
if [ -z "${GPUS_PER_NODE:-}" ]; then
    GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
    [ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null || GPUS_PER_NODE=8
fi
NODE_IP="${NODE_IP:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-6379}"

mkdir -p "${OUTPUT_DIR}"
ray stop >/dev/null 2>&1 || true
ray start --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --num-gpus="${GPUS_PER_NODE}"

exec "${CMD[@]}"
