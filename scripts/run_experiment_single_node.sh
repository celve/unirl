#!/usr/bin/env bash
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
CMD+=("$@")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

default_node_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || true
}

GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
NODE_IP="${NODE_IP:-${LOCAL_IP:-$(default_node_ip)}}"
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
