#!/bin/bash
# =============================================================================
# DiffusionNFT SD3 multi-node — SGLang separate mode
# =============================================================================
#
# NFT + SGLang rollout engine. NFT is a forward-process algorithm:
# ``NFTAlgorithm.resolve_rollout_sde_indices`` returns None (no SDE rollout
# indices), so SGLang runs ODE sampling to produce clean final latents.  NFT
# then trains on (clean_latent, reward, prompt_id) tuples using a forward
# diffusion loss — no per-step log_probs are needed from the rollout side,
# which is why ``--sampling.logprob-source`` is irrelevant here (we keep the
# ``replay`` default purely for symmetry with the GRPO-SGLang script and to
# avoid SGLang trying to emit log_probs it would never consume).
#
# Cluster platform env auto-detection (Taiji/Jizhi):
#   CHIEF_IP     -> HEAD_IP       INDEX        -> ROLE (0=head, >0=worker)
#   LOCAL_IP     -> NODE_IP       HOST_NUM     -> NUM_NODES
#   HOST_GPU_NUM -> GPUS_PER_NODE
#
# Usage:
#   # Auto mode (Taiji/Jizhi platform, each node runs the same command):
#   bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh
#   bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh auto
#
#   # Manual mode:
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.1 bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh head
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.2 bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh worker
#   HEAD_IP=10.0.0.1 bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh train
#
#   # Pass through extra diffusionrl CLI overrides:
#   bash reproduce_scripts/train_nft_sd3_sglang_multinode.sh auto \
#       --rollout.num-rollout 100 --training.micro-batch-size 6
#
# Key alignment with original DiffusionNFT:
#   algorithm_type=nft, beta=1.0, kl_coef=0.0001, sde_type=dpm2,
#   num_inference_steps=10, guidance_scale=4.5, adv_normalization=group,
#   train_timestep_mode=all, LoRA rank=32 alpha=64
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
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/nft_sd3_sglang_multinode}"
WORKER_WAIT_INTERVAL="${WORKER_WAIT_INTERVAL:-10}"
WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-600}"

# ── SGLang configuration ──
# NFT doesn't consume rollout-side log_probs (forward-process training), so
# logprob_source is effectively a no-op here; kept at ``replay`` for parity
# with the GRPO-SGLang script so operators don't see surprising diffs.
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-4}"
TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-4}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/nft_sd3_sglang_multinode}"
SDE_TYPE="${SDE_TYPE:-dpm2}"
TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-0.99}"
REWARD_NAME="${REWARD_NAME:-pickscore}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/${REWARD_NAME}/train.txt}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/${REWARD_NAME}/test.txt}"
EVAL_STEPS="${EVAL_STEPS:-0}"

# ── Batch geometry (5 core knobs — see _batch_config.sh for docs) ──
# For SGLang separate mode, batch geometry is sized against the *training*
# data-parallel world (rollout GPUs never see a TrainingBatch), which equals
# NUM_NODES * TRAINING_GPUS_PER_NODE. Temporarily pass TRAINING_GPUS_PER_NODE
# as GPUS_PER_NODE into resolve_batch_params, then restore the physical count.
NUM_INFERENCE_STEPS=10
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-48}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-16}
# SGLang routes sampling through --rollout.rollout-batch-size, not
# --sampling.max-samples-per-request (which is ignored by the SGLang engine).
SAMPLING_FORWARD_BATCH=${SAMPLING_FORWARD_BATCH:-${NUM_SAMPLES_PER_PROMPT}}
TRAINING_FORWARD_BATCH=${TRAINING_FORWARD_BATCH:-8}
NUM_UPDATES=${NUM_UPDATES:-1}

_BATCH_GPUS_PER_NODE_SAVED="${GPUS_PER_NODE}"
GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE}"
source "${REPO_ROOT}/scripts/_batch_config.sh"
resolve_batch_params
validate_batch_params
print_batch_params
GPUS_PER_NODE="${_BATCH_GPUS_PER_NODE_SAVED}"


EVAL_EMA_DECAY="${EVAL_EMA_DECAY:-0.99}"
EVAL_EMA_UPDATE_INTERVAL="${EVAL_EMA_UPDATE_INTERVAL:-1}"

REPORT_TO_WANDB=true
WANDB_PROJECT_NAME="diffusionrl-diffusionNFT"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-DiffusionNFT-SGLang-multinode}"
WANDB_LOG_MEDIA=true
WANDB_MEDIA_MAX_ITEMS=48
WANDB_TAGS="${WANDB_TAGS:-reproduce,sd3.5,nft,sglang,${REWARD_NAME},multinode}"
WANDB_ENTITY="${WANDB_ENTITY:-diffusionrl-reproduce}"
LOGGING_STEPS=1

REWARD_DEVICE="cuda"

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
    echo "Topology: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs (${TRAINING_GPUS_PER_NODE} train + ${ROLLOUT_GPUS_PER_NODE} rollout per node)"
    echo "Weight sync dir: ${WEIGHT_SYNC_DIR}"
    echo "Output dir: ${OUTPUT_DIR}"

    check_wandb_auth

    python -m diffusionrl.train \
        --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
        --model.model-type sd3 \
        --algorithm.algorithm-dotpath diffusionrl.algorithms.nft.NFTAlgorithm \
        --reward.reward-components "${REWARD_NAME}" \
        --reward.local-reward-device "${REWARD_DEVICE}" \
        --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
        --data-path "${DATA_PATH}" \
        --eval-data-path "${EVAL_DATA_PATH}" \
        \
        --rollout.mode separate \
        --rollout.rollout-engine sglang \
        --rollout.num-gpus-per-actor "${TP_SIZE}" \
        --rollout.tp-size "${TP_SIZE}" \
        --rollout.rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
        \
        --sampling.shift 3.0 \
        --sampling.eta 0.0 \
        --sampling.sde-type "${SDE_TYPE}" \
        --algorithm.training-scheduler.timestep-fraction "${TIMESTEP_FRACTION}" \
        --sampling.num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --sampling.guidance-scale 1.0 \
        --algorithm.kwarg beta=1 \
        --algorithm.kwarg adv_mode=raw \
        --algorithm.kwarg adv_clip_max=5.0 \
        --algorithm.kwarg use_adaptive_weight=true \
        --algorithm.kwarg train_timestep_mode=all \
        --algorithm.kwarg shuffle_train_timesteps=true \
        --algorithm.kwarg apply_time_shift_in_loss=false \
        --algorithm.kwarg use_reference_ema=true \
        --algorithm.kwarg reference_update_timing=rollout_end \
        --algorithm.kwarg ema_decay=0.001 \
        --algorithm.kwarg ema_decay_type=warmup \
        --algorithm.kwarg ema_flat_steps=0 \
        --algorithm.kwarg ema_uprate=0.001 \
        --algorithm.kwarg ema_uphold=0.5 \
        \
        --algorithm.prompts-per-rollout "${PROMPTS_PER_BATCH}" \
        "${micro_batch_args[@]}" \
        --training.num-updates-per-batch "${NUM_UPDATES_PER_BATCH}" \
        --algorithm.samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}" \
        --algorithm.kwarg kl_coef=0.0 \
        --algorithm.adv-normalization group \
        --algorithm.use-global-std true \
        --algorithm.adv-norm-eps 1e-4 \
        --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}" \
        --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}" \
        \
        --sync.protocol nccl_broadcast \
        --sync.dir "${WEIGHT_SYNC_DIR}" \
        --ray.rollout-num-nodes "${NUM_NODES}" \
        --ray.rollout-num-gpus-per-node "${ROLLOUT_GPUS_PER_NODE}" \
        --ray.training-num-gpus-per-node "${TRAINING_GPUS_PER_NODE}" \
        --ray.ray-address "${HEAD_IP}:${HEAD_PORT}" \
        --ray.training-num-nodes "${NUM_NODES}" \
        --ray.placement-strategy "${RAY_PLACEMENT_STRATEGY}" \
        --ray.offload-train false \
        --ray.offload-rollout false \
        \
        --training.learning-rate 3e-4 \
        --training.max-grad-norm 1.0 \
        --training.lora-rank 32 \
        --training.lora-alpha 64 \
        --training.use-lora true \
        --training.use-gradient-checkpointing false \
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
        "$@"
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

    stop)   ray stop >/dev/null 2>&1 || true; echo "Ray stopped on local node." ;;
    status) if [ -n "${HEAD_IP}" ]; then ray status --address "${HEAD_IP}:${HEAD_PORT}"; else ray status; fi ;;
    *)      echo "ERROR: unknown role: ${ROLE}"; print_usage; exit 1 ;;
esac
