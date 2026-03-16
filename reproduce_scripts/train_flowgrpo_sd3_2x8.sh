#!/bin/bash
# =============================================================================
# FlowGRPO SD3 multi-node helper (2 nodes x 8 GPUs by default)
# =============================================================================
#
# This helper script orchestrates:
#   1) Ray head startup
#   2) Ray worker join
#   3) Training submission (from head node)
#
# It reuses:
#   reproduce_scripts/train_flowgrpo_sd3_train_actor_sampling_pickscore.sh
#
# Usage examples:
#   # Node 0 (head)
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.1 bash reproduce_scripts/train_flowgrpo_sd3_2x8.sh head
#
#   # Node 1 (worker)
#   HEAD_IP=10.0.0.1 NODE_IP=10.0.0.2 bash reproduce_scripts/train_flowgrpo_sd3_2x8.sh worker
#
#   # Submit training on head node
#   HEAD_IP=10.0.0.1 WEIGHT_SYNC_DIR=/mnt/shared/diffusionrl_weight_sync \
#     bash reproduce_scripts/train_flowgrpo_sd3_2x8.sh train
#
#   # Pass through extra diffusionrl CLI overrides
#   HEAD_IP=10.0.0.1 WEIGHT_SYNC_DIR=/mnt/shared/diffusionrl_weight_sync \
#     bash reproduce_scripts/train_flowgrpo_sd3_2x8.sh train \
#       --rollout.num-rollout 100 --training.gradient-accumulation-batch-size 6
#
# =============================================================================

set -euo pipefail

# NCCL environment variables for multi-node IB/RDMA
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
BASE_TRAIN_SCRIPT="${SCRIPT_DIR}/train_flowgrpo_sd3_train_actor_sampling_pickscore.sh"

ROLE="${1:-}"
if [ -n "${ROLE}" ]; then
    shift
fi

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

HEAD_IP="${HEAD_IP:-}"
NODE_IP="${NODE_IP:-}"
HEAD_PORT="${HEAD_PORT:-6379}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
RAY_PLACEMENT_STRATEGY="${RAY_PLACEMENT_STRATEGY:-SPREAD}"
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/mnt/shared/diffusionrl_weight_sync/flowgrpo_sd3_2x8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_sd3_2x8}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-Flow-GRPO-2x8}"
WANDB_TAGS="${WANDB_TAGS:-reproduce,sd3.5,flow,pickscore,2x8}"

print_usage() {
    cat <<EOF
Usage:
  $(basename "$0") head
  $(basename "$0") worker
  $(basename "$0") train [diffusionrl args...]
  $(basename "$0") stop
  $(basename "$0") status

Required env by role:
  head:
    HEAD_IP=<head node ip>
    NODE_IP=<current node ip>         # optional, defaults to HEAD_IP

  worker:
    HEAD_IP=<head node ip>
    NODE_IP=<current node ip>

  train:
    HEAD_IP=<head node ip>
    WEIGHT_SYNC_DIR=<shared fs path>  # required for multi-node checkpoint sync

Common env:
  NUM_NODES=2
  GPUS_PER_NODE=8
  HEAD_PORT=6379
  RAY_PLACEMENT_STRATEGY=SPREAD
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: command not found: ${cmd}"
        exit 1
    fi
}

require_file() {
    local f="$1"
    if [ ! -f "${f}" ]; then
        echo "ERROR: file not found: ${f}"
        exit 1
    fi
}

if [ -z "${ROLE}" ]; then
    print_usage
    exit 1
fi

require_cmd ray
require_file "${BASE_TRAIN_SCRIPT}"

case "${ROLE}" in
    head)
        if [ -z "${HEAD_IP}" ]; then
            echo "ERROR: HEAD_IP is required for role=head"
            exit 1
        fi
        if [ -z "${NODE_IP}" ]; then
            NODE_IP="${HEAD_IP}"
        fi
        ray stop >/dev/null 2>&1 || true
        echo "Starting Ray head on ${NODE_IP} (port=${HEAD_PORT}, gpus=${GPUS_PER_NODE})"
        ray start \
            --head \
            --node-ip-address "${NODE_IP}" \
            --port "${HEAD_PORT}" \
            --dashboard-host "${DASHBOARD_HOST}" \
            --num-gpus "${GPUS_PER_NODE}"
        ;;

    worker)
        if [ -z "${HEAD_IP}" ] || [ -z "${NODE_IP}" ]; then
            echo "ERROR: HEAD_IP and NODE_IP are required for role=worker"
            exit 1
        fi
        ray stop >/dev/null 2>&1 || true
        echo "Joining Ray cluster ${HEAD_IP}:${HEAD_PORT} from ${NODE_IP} (gpus=${GPUS_PER_NODE})"
        ray start \
            --address "${HEAD_IP}:${HEAD_PORT}" \
            --node-ip-address "${NODE_IP}" \
            --num-gpus "${GPUS_PER_NODE}"
        ;;

    train)
        if [ -z "${HEAD_IP}" ]; then
            echo "ERROR: HEAD_IP is required for role=train"
            exit 1
        fi
        mkdir -p "${WEIGHT_SYNC_DIR}"
        echo "Submitting training to Ray cluster ${HEAD_IP}:${HEAD_PORT}"
        echo "Topology: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs"
        echo "Weight sync dir: ${WEIGHT_SYNC_DIR}"
        echo "Output dir: ${OUTPUT_DIR}"
        bash "${BASE_TRAIN_SCRIPT}" \
            --ray.ray-address "${HEAD_IP}:${HEAD_PORT}" \
            --ray.training-num-nodes "${NUM_NODES}" \
            --ray.training-num-gpus-per-node "${GPUS_PER_NODE}" \
            --ray.placement-strategy "${RAY_PLACEMENT_STRATEGY}" \
            --ray.weight-sync-dir "${WEIGHT_SYNC_DIR}" \
            --rollout.run-name "${WANDB_RUN_NAME}" \
            --rollout.output-dir "${OUTPUT_DIR}" \
            --rollout.wandb-tags "${WANDB_TAGS}" \
            "$@"
        ;;

    stop)
        ray stop >/dev/null 2>&1 || true
        echo "Ray stopped on local node."
        ;;

    status)
        if [ -n "${HEAD_IP}" ]; then
            ray status --address "${HEAD_IP}:${HEAD_PORT}"
        else
            ray status
        fi
        ;;

    *)
        echo "ERROR: unknown role: ${ROLE}"
        print_usage
        exit 1
        ;;
esac
