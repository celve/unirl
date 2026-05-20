#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Optional environment activation. Leave CONDA_ENV unset if the caller already
# prepared Python, CUDA, and dependencies.
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
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

ray stop >/dev/null 2>&1 || true

NUM_NODES=2
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
RAY_PORT="${RAY_PORT:-6379}"
NODE_IP="${NODE_IP:-${LOCAL_IP:-$(hostname -I | awk '{print $1}')}}"
HEAD_IP="${HEAD_IP:-${CHIEF_IP:-${NODE_IP}}}"

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

# Give workers time to join before the placement group requests all GPUs.
sleep "${RAY_CLUSTER_WAIT_S:-30}"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_fast_qwen_image_2x8}"
QWEN_IMAGE_PATH="${QWEN_IMAGE_PATH:-Qwen/Qwen-Image}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-Qwen-Image-FlowGRPO-2x8}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"

CMD=(
    python -m diffusionrl.train_new
    +experiment=flowgrpo_fast_qwen_image_new
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "algorithm.samples_per_prompt=${SAMPLES_PER_PROMPT:-12}"
    "placement.num_rollout_nodes=${NUM_NODES}"
    "placement.num_rollout_gpus_per_node=${GPUS_PER_NODE}"
    "placement.num_train_nodes=${NUM_NODES}"
    "placement.num_train_gpus_per_node=${GPUS_PER_NODE}"
    "training.plan.global_batch_size=${GLOBAL_BATCH_SIZE:-576}"
    "training.plan.local_mini_batch_size=${LOCAL_MINI_BATCH_SIZE:-18}"
    "training.plan.local_batch_size=${LOCAL_BATCH_SIZE:-36}"
    "training.topology.actor_count=${ACTOR_COUNT:-16}"
    "sampling.height=${HEIGHT:-256}"
    "sampling.width=${WIDTH:-256}"
    "rollout.plan.forward_batch_size=${FORWARD_BATCH_SIZE:-1}"
    "model.pretrained_model_ckpt_path=${QWEN_IMAGE_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
    "logging.run_name=${WANDB_RUN_NAME}"
    "logging.report_to_wandb=${REPORT_TO_WANDB}"
)
CMD+=("$@")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
