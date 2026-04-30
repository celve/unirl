#!/bin/bash
# =============================================================================
# FlowGRPO SD3 multi-node — SGLang colocate mode + TransferQueue (Mooncake)
# =============================================================================
#
# Colocate variant of reproduce_scripts/train_grpo_sd3_sglang_multinode.sh:
# rollout and training share the same GPU bundles via fractional claims;
# sglang rollout is offloaded during the training phase. Tensors flow between
# rollout and train actors via TransferQueue (mooncake RDMA zero-copy).
#
# Cluster platform env auto-detection (Taiji/Jizhi):
#   CHIEF_IP     -> HEAD_IP       INDEX        -> ROLE (0=head, >0=worker)
#   LOCAL_IP     -> NODE_IP       HOST_NUM     -> NUM_NODES
#   HOST_GPU_NUM -> GPUS_PER_NODE
#
# Usage:
#   # Auto mode (Taiji/Jizhi platform, each node runs the same command):
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh auto
#
#   # Manual mode:
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.1 \
#     bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh head
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.2 \
#     bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh worker
#   HEAD_IP=10.0.0.1 \
#     bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh train
#
#   # Pass through extra diffusionrl CLI overrides:
#   bash reproduce_scripts/train_grpo_sd3_sglang_colocate_tq_multinode.sh auto \
#       --rollout.num-rollout 100 --training.micro-batch-size 6
#
# Key alignment with FlowGRPO:
#   sde_type=flow, eta=0.7, shift=3.0, num_inference_steps=10,
#   guidance_scale=1.0, kl_coef=0.0, adv_normalization_scope=group,
#   learning_rate=3e-4, LoRA rank=32 alpha=64, timestep_fraction=[0.0,0.5]
#
# Colocate+TQ specifics:
#   rollout.mode=colocate, rollout-engine=sglang, tp_size=1,
#   num_gpus_per_actor=1, sync.protocol=tensor_payload, offload-rollout=true,
#   transfer-queue.enable=true (Mooncake backend with RDMA zero-copy).
#
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
source "${REPO_ROOT}/scripts/_check_wandb.sh"

# ── Role handling ──
ROLE="${1:-}"
if [ -n "${ROLE}" ]; then shift; fi

if [ -z "${ROLE}" ] || [ "${ROLE}" = "auto" ]; then
    if [ -n "${INDEX:-}" ]; then
        if [ "${INDEX}" = "0" ]; then ROLE="auto_head"; else ROLE="auto_worker"; fi
    fi
fi

# ── Cluster platform env auto-detection (Taiji/Jizhi) ──
NUM_NODES="${NUM_NODES:-${HOST_NUM:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
HEAD_IP="${HEAD_IP:-${CHIEF_IP:-}}"
NODE_IP="${NODE_IP:-${LOCAL_IP:-}}"
HEAD_PORT="${HEAD_PORT:-6379}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
RAY_PLACEMENT_STRATEGY="${RAY_PLACEMENT_STRATEGY:-SPREAD}"
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/grpo_sd3_sglang_colocate_multinode}"
WORKER_WAIT_INTERVAL="${WORKER_WAIT_INTERVAL:-10}"
WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-600}"

# ── SGLang configuration ──
TP_SIZE="${TP_SIZE:-1}"
SGLANG_LOGPROB_MODE="${SGLANG_LOGPROB_MODE:-replay}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"

# Prefer local sibling sglang checkout when available; otherwise use installed package.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi

# ── Training hyperparameters ──
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-3.5-medium}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo_sd3_sglang_colocate_multinode}"
SDE_TYPE="${SDE_TYPE:-flow}"
TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-[0.0,0.5]}"
NUM_SDE_STEPS="${NUM_SDE_STEPS:-3}"
REWARD_NAME="${REWARD_NAME:-pickscore}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/${REWARD_NAME}/train.txt}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/${REWARD_NAME}/test.txt}"
EVAL_STEPS="${EVAL_STEPS:-0}"

# ── Batch geometry (5 core knobs — see _batch_config.sh for docs) ──
NUM_INFERENCE_STEPS=10
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-48}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-16}
SAMPLING_FORWARD_BATCH=${SAMPLING_FORWARD_BATCH:-${NUM_SAMPLES_PER_PROMPT}}  # passthrough to --sampling.forward-batch-size
TRAINING_FORWARD_BATCH=${TRAINING_FORWARD_BATCH:-8}     # per-device peak forward batch during training
NUM_UPDATES=${NUM_UPDATES:-2}                           # gradient update steps per local batch

source "${REPO_ROOT}/scripts/_batch_config.sh"
resolve_batch_params
validate_batch_params
print_batch_params


EVAL_EMA_DECAY="${EVAL_EMA_DECAY:-0.99}"
EVAL_EMA_UPDATE_INTERVAL="${EVAL_EMA_UPDATE_INTERVAL:-2}"

REPORT_TO_WANDB=true
WANDB_PROJECT_NAME="diffusionrl-flowgrpo"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-Flow-GRPO-SGLang-colocate-TQ-multinode}"
WANDB_LOG_MEDIA=true
WANDB_MEDIA_MAX_ITEMS=48
WANDB_TAGS="${WANDB_TAGS:-reproduce,sd3.5,flow,sglang,${REWARD_NAME},colocate,transfer-queue,multinode}"
WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
LOGGING_STEPS=1

REWARD_DEVICE="cuda"

# ── TransferQueue / Mooncake configuration ──
TRANSFER_QUEUE_UTILS_DEBUG="${TRANSFER_QUEUE_UTILS_DEBUG:-0}"
TQ_LOGGING_LEVEL="${TQ_LOGGING_LEVEL:-WARNING}"
MC_YLT_LOG_LEVEL="${MC_YLT_LOG_LEVEL:-warning}"
TQ_ENABLE="${TQ_ENABLE:-true}"
TQ_MC_METADATA_SERVER_PORT="${TQ_MC_METADATA_SERVER_PORT:-50041}"
TQ_MC_RPC_PORT="${TQ_MC_RPC_PORT:-50051}"
TQ_STORAGE_BACKEND="${TQ_STORAGE_BACKEND:-MooncakeStorageManager}"
TQ_MC_METADATA_SERVER="${TQ_MC_METADATA_SERVER:-http://${HEAD_IP}:${TQ_MC_METADATA_SERVER_PORT}/metadata}"
TQ_MC_MASTER_SERVER_ADDR="${TQ_MC_MASTER_SERVER_ADDR:-${HEAD_IP}:${TQ_MC_RPC_PORT}}"
TQ_MC_GLOBAL_SEGMENT_SIZE_GB="${TQ_MC_GLOBAL_SEGMENT_SIZE_GB:-10}"
TQ_MC_LOCAL_BUFFER_SIZE_GB="${TQ_MC_LOCAL_BUFFER_SIZE_GB:-10}"
TQ_MC_DEVICE_NAME="${TQ_MC_DEVICE_NAME:-mlx5_bond_1}"
TQ_MC_ZERO_COPY_ENABLE="${TQ_MC_ZERO_COPY_ENABLE:-true}"
TQ_MC_ZERO_COPY_TENSOR_BUFFER_SIZE_GB="${TQ_MC_ZERO_COPY_TENSOR_BUFFER_SIZE_GB:-2}"
TQ_MC_ZERO_COPY_BYTES_BUFFER_SIZE_GB="${TQ_MC_ZERO_COPY_BYTES_BUFFER_SIZE_GB:-2}"
TQ_MC_ZERO_COPY_SCTRL_TENSOR_BUFFER_SIZE_GB="${TQ_MC_ZERO_COPY_SCTRL_TENSOR_BUFFER_SIZE_GB:-10}"
TQ_MC_ZERO_COPY_SCTRL_BYTES_BUFFER_SIZE_GB="${TQ_MC_ZERO_COPY_SCTRL_BYTES_BUFFER_SIZE_GB:-10}"
TQ_MC_ZERO_COPY_MANAGER_MERGE_TO_TENSORDICT="${TQ_MC_ZERO_COPY_MANAGER_MERGE_TO_TENSORDICT:-false}"

# ── Helpers ──
print_usage() {
    cat <<EOF
Usage:
  $(basename "$0")                             # auto mode (uses INDEX env)
  $(basename "$0") auto [diffusionrl args...]  # same as above, explicit
  $(basename "$0") head
  $(basename "$0") worker
  $(basename "$0") train [diffusionrl args...]
  $(basename "$0") stop
  $(basename "$0") status

Cluster env auto-detection (Taiji/Jizhi platform):
  CHIEF_IP     -> HEAD_IP       INDEX        -> ROLE (0=head, >0=worker)
  LOCAL_IP     -> NODE_IP       HOST_NUM     -> NUM_NODES
  HOST_GPU_NUM -> GPUS_PER_NODE
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then echo "ERROR: command not found: $1"; exit 1; fi
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

run_mooncake() {
    if [[ "$TQ_ENABLE" == "true" && "$TQ_STORAGE_BACKEND" == "MooncakeStorageManager" ]]; then
        mooncake_log_file=$(mktemp /tmp/mooncake.log.XXXXXX)
        echo "Mooncake log put to ${mooncake_log_file}"
        kill_mooncake
        if lsof -i :${TQ_MC_METADATA_SERVER_PORT} > /dev/null 2>&1; then
            echo -e "\033[31m${TQ_MC_METADATA_SERVER_PORT} port in use! kill process or change port!\033[0m"
            exit 1
        fi
        if lsof -i :${TQ_MC_RPC_PORT} > /dev/null 2>&1; then
            echo -e "\033[31m${TQ_MC_RPC_PORT} port in use! kill process or change port!\033[0m"
            exit 1
        fi
        nohup mooncake_master \
            --enable_http_metadata_server=true \
            --http_metadata_server_host=0.0.0.0 \
            --http_metadata_server_port=${TQ_MC_METADATA_SERVER_PORT} \
            --rpc_port=${TQ_MC_RPC_PORT} \
            --rpc_thread_num=64 \
            --default_kv_lease_ttl=100000000 > "${mooncake_log_file}" 2>&1 &
    fi
}

kill_mooncake() {
    pkill -x mooncake_master || true
}

# ── Training command ──
run_training() {

    local micro_batch_args=()
    if [ -n "${MICRO_BATCH_SIZE}" ]; then
        micro_batch_args+=(--training.micro-batch-size "${MICRO_BATCH_SIZE}")
    fi

    local wandb_entity_args=()
    if [ -n "${WANDB_ENTITY}" ]; then
        wandb_entity_args+=(--logging.entity "${WANDB_ENTITY}")
    fi

    mkdir -p "${OUTPUT_DIR}"
    echo "Submitting training to Ray cluster ${HEAD_IP}:${HEAD_PORT}"
    echo "Topology: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs (colocate sglang + TransferQueue)"
    echo "Weight sync dir: ${WEIGHT_SYNC_DIR}"
    echo "Output dir: ${OUTPUT_DIR}"

    check_wandb_auth

    run_mooncake

    python -m diffusionrl.train \
        --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
        --model.model-type sd3 \
        --algorithm.algorithm-dotpath diffusionrl.algorithms.grpo.GRPOAlgorithm \
        --reward.reward-components "${REWARD_NAME}" \
        --reward.local-reward-device "${REWARD_DEVICE}" \
        --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
        --data-path "${DATA_PATH}" \
        --eval-data-path "${EVAL_DATA_PATH}" \
        \
        --rollout.mode colocate \
        --rollout.rollout-engine sglang \
        --rollout.num-gpus-per-actor "${TP_SIZE}" \
        --rollout.tp-size "${TP_SIZE}" \
        --rollout.sglang-disable-autocast true \
        --sampling.forward-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
        --sampling.init-same-noise true \
        \
        --sampling.sde-type "${SDE_TYPE}" \
        --sampling.eta 0.7 \
        --sampling.shift 3.0 \
        --sampling.num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --sampling.guidance-scale 1.0 \
        --algorithm.rollout-scheduler.timestep-fraction "${TIMESTEP_FRACTION}" \
        --algorithm.rollout-scheduler.num-sde-steps "${NUM_SDE_STEPS}" \
        \
        --algorithm.prompts-per-rollout "${PROMPTS_PER_BATCH}" \
        "${micro_batch_args[@]}" \
        --training.num-updates-per-batch "${NUM_UPDATES_PER_BATCH}" \
        --algorithm.samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}" \
        --algorithm.kwarg clip_range=1e-4 \
        --algorithm.kwarg use_kl_penalty=true \
        --algorithm.kwarg kl_coef=0.0 \
        --algorithm.adv-normalization-scope group \
        --algorithm.use-global-std true \
        --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}" \
        --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}" \
        \
        --sync.protocol tensor_payload \
        --sync.dir "${WEIGHT_SYNC_DIR}" \
        --ray.rollout-num-nodes "${NUM_NODES}" \
        --ray.rollout-num-gpus-per-node "${GPUS_PER_NODE}" \
        --ray.training-num-gpus-per-node "${GPUS_PER_NODE}" \
        --ray.ray-address "${HEAD_IP}:${HEAD_PORT}" \
        --ray.training-num-nodes "${NUM_NODES}" \
        --ray.placement-strategy "${RAY_PLACEMENT_STRATEGY}" \
        --ray.offload-rollout true \
        --ray.offload-train false \
        \
        --training.learning-rate 3e-4 \
        --training.max-grad-norm 1.0 \
        --training.lora-rank 32 \
        --training.lora-alpha 64 \
        --training.use-lora true \
        \
        --sampling.height 512 \
        --sampling.width 512 \
        \
        --rollout.num-rollout 10000 \
        --rollout.save-steps 0 \
        --evaluation.eval-steps ${EVAL_STEPS} \
        --logging.logging-steps "${LOGGING_STEPS}" \
        --rollout.output-dir "${OUTPUT_DIR}" \
        --logging.report-to-wandb "${REPORT_TO_WANDB}" \
        --logging.project-name "${WANDB_PROJECT_NAME}" \
        --logging.run-name "${WANDB_RUN_NAME}" \
        --logging.log-media "${WANDB_LOG_MEDIA}" \
        --logging.media-max-items "${WANDB_MEDIA_MAX_ITEMS}" \
        --logging.tags "${WANDB_TAGS}" \
        "${wandb_entity_args[@]}" \
        \
        --transfer-queue.enable "${TQ_ENABLE}" \
        --transfer-queue.storage-backend "${TQ_STORAGE_BACKEND}" \
        --transfer-queue.mooncake-backend-config.metadata-server "${TQ_MC_METADATA_SERVER}" \
        --transfer-queue.mooncake-backend-config.master-server-address "${TQ_MC_MASTER_SERVER_ADDR}" \
        --transfer-queue.mooncake-backend-config.global-segment-size-gb "${TQ_MC_GLOBAL_SEGMENT_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.local-buffer-size-gb "${TQ_MC_LOCAL_BUFFER_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.client-name MooncakeStorageClient \
        --transfer-queue.mooncake-backend-config.protocol rdma \
        --transfer-queue.mooncake-backend-config.device-name "${TQ_MC_DEVICE_NAME}" \
        --transfer-queue.mooncake-backend-config.zero-copy-enable "${TQ_MC_ZERO_COPY_ENABLE}" \
        --transfer-queue.mooncake-backend-config.zero-copy-tensor-buffer-size-gb "${TQ_MC_ZERO_COPY_TENSOR_BUFFER_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.zero-copy-bytes-buffer-size-gb "${TQ_MC_ZERO_COPY_BYTES_BUFFER_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.zero-copy-single-controller-tensor-buffer-size-gb "${TQ_MC_ZERO_COPY_SCTRL_TENSOR_BUFFER_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.zero-copy-single-controller-bytes-buffer-size-gb "${TQ_MC_ZERO_COPY_SCTRL_BYTES_BUFFER_SIZE_GB}" \
        --transfer-queue.mooncake-backend-config.zero-copy-manager-merge-to-tensordict "${TQ_MC_ZERO_COPY_MANAGER_MERGE_TO_TENSORDICT}" \
        "$@"

    kill_mooncake
}

# ── Main ──
if [ -z "${ROLE}" ]; then print_usage; exit 1; fi
require_cmd ray

case "${ROLE}" in
    auto_head)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP (or CHIEF_IP) is required"; exit 1; fi
        : "${NODE_IP:=${HEAD_IP}}"
        ray stop >/dev/null 2>&1 || true
        echo "[INDEX=${INDEX:-0}] Starting Ray head on ${NODE_IP} (port=${HEAD_PORT}, gpus=${GPUS_PER_NODE})"
        ray start --head --node-ip-address "${NODE_IP}" --port "${HEAD_PORT}" \
            --dashboard-host "${DASHBOARD_HOST}" --num-gpus "${GPUS_PER_NODE}"
        wait_for_workers "${NUM_NODES}"
        mkdir -p "${WEIGHT_SYNC_DIR}"
        run_training "$@"
        ;;

    auto_worker)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP (or CHIEF_IP) is required"; exit 1; fi
        if [ -z "${NODE_IP}" ]; then echo "ERROR: NODE_IP (or LOCAL_IP) is required for worker"; exit 1; fi
        ray stop >/dev/null 2>&1 || true
        echo "[INDEX=${INDEX:-?}] Joining Ray cluster ${HEAD_IP}:${HEAD_PORT} from ${NODE_IP} (gpus=${GPUS_PER_NODE})"
        ray start --address "${HEAD_IP}:${HEAD_PORT}" --node-ip-address "${NODE_IP}" --num-gpus "${GPUS_PER_NODE}"
        echo "Worker joined. Keeping process alive..."
        tail -f /dev/null
        ;;

    head)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP is required for role=head"; exit 1; fi
        : "${NODE_IP:=${HEAD_IP}}"
        ray stop >/dev/null 2>&1 || true
        echo "Starting Ray head on ${NODE_IP} (port=${HEAD_PORT}, gpus=${GPUS_PER_NODE})"
        ray start --head --node-ip-address "${NODE_IP}" --port "${HEAD_PORT}" \
            --dashboard-host "${DASHBOARD_HOST}" --num-gpus "${GPUS_PER_NODE}"
        ;;

    worker)
        if [ -z "${HEAD_IP}" ] || [ -z "${NODE_IP}" ]; then
            echo "ERROR: HEAD_IP and NODE_IP are required for role=worker"; exit 1
        fi
        ray stop >/dev/null 2>&1 || true
        echo "Joining Ray cluster ${HEAD_IP}:${HEAD_PORT} from ${NODE_IP} (gpus=${GPUS_PER_NODE})"
        ray start --address "${HEAD_IP}:${HEAD_PORT}" --node-ip-address "${NODE_IP}" --num-gpus "${GPUS_PER_NODE}"
        ;;

    train)
        if [ -z "${HEAD_IP}" ]; then echo "ERROR: HEAD_IP is required for role=train"; exit 1; fi
        mkdir -p "${WEIGHT_SYNC_DIR}"
        run_training "$@"
        ;;

    stop)   ray stop >/dev/null 2>&1 || true; kill_mooncake; echo "Ray + mooncake stopped on local node." ;;
    status) if [ -n "${HEAD_IP}" ]; then ray status --address "${HEAD_IP}:${HEAD_PORT}"; else ray status; fi ;;
    *)      echo "ERROR: unknown role: ${ROLE}"; print_usage; exit 1 ;;
esac
