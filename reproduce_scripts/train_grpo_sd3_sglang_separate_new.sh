#!/bin/bash
# =============================================================================
# FlowGRPO SD3 — SGLang rollout engine, separate mode (NEW-DESIGN path)
#
# Supports both native and replay logprob modes via LOGPROB_SOURCE env var.
#
# Usage:
#   # Native mode (SGLang computes logprobs during inference):
#   LOGPROB_SOURCE=native bash reproduce_scripts/train_grpo_sd3_sglang_separate_new.sh
#
#   # Replay mode (trainer replays forward to compute logprobs):
#   LOGPROB_SOURCE=replay bash reproduce_scripts/train_grpo_sd3_sglang_separate_new.sh
#
#   # Default is native mode
#   bash reproduce_scripts/train_grpo_sd3_sglang_separate_new.sh
#
# Prerequisites:
#   - CHIEF_IP, NODE_IP_LIST set by platform (2-node cluster)
#   - WANDB_API_KEY set
#   - HF_TOKEN set (for model download)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# ★ 用户配置区
# ============================================================
NODE_COUNT="${HOST_NUM:-2}"
RANK_PER_NODE="${HOST_GPU_NUM:-8}"

# Logprob mode: "native" or "replay"
LOGPROB_SOURCE="${LOGPROB_SOURCE:-native}"
# Placement mode: "separate" or "colocate"
PLACEMENT_MODE="${PLACEMENT_MODE:-separate}"

if [ "${LOGPROB_SOURCE}" = "native" ] && [ "${PLACEMENT_MODE}" = "separate" ]; then
    EXPERIMENT="flowgrpo_sd3_sglang_native_separate"
elif [ "${LOGPROB_SOURCE}" = "replay" ] && [ "${PLACEMENT_MODE}" = "separate" ]; then
    EXPERIMENT="flowgrpo_sd3_sglang_replay_separate"
elif [ "${LOGPROB_SOURCE}" = "native" ] && [ "${PLACEMENT_MODE}" = "colocate" ]; then
    EXPERIMENT="flowgrpo_sd3_sglang_native_colocate"
elif [ "${LOGPROB_SOURCE}" = "replay" ] && [ "${PLACEMENT_MODE}" = "colocate" ]; then
    EXPERIMENT="flowgrpo_sd3_sglang_replay_colocate"
else
    echo "[ERROR] Unsupported combination: LOGPROB_SOURCE=${LOGPROB_SOURCE}, PLACEMENT_MODE=${PLACEMENT_MODE}"
    exit 1
fi

export PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-3.5-medium}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXPERIMENT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-SD3.5-FlowGRPO-sglang-${LOGPROB_SOURCE}-${PLACEMENT_MODE}}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-${REPO_ROOT}/outputs/weight_sync/${EXPERIMENT}}"

HEAD_PORT="${HEAD_PORT:-6379}"
SSH_USER="${SSH_USER:-$(whoami)}"
CHIEF_IP="${CHIEF_IP:?ERROR: CHIEF_IP env var must be set}"
NODE_IP_LIST="${NODE_IP_LIST:?ERROR: NODE_IP_LIST env var must be set}"

# ============================================================
# ★ 解析 NODE_IP_LIST
# ============================================================
IFS=',' read -ra NODE_ENTRIES <<< "${NODE_IP_LIST}"
if [ ${#NODE_ENTRIES[@]} -lt ${NODE_COUNT} ]; then
    echo "[ERROR] NODE_IP_LIST 中的节点数 (${#NODE_ENTRIES[@]}) 少于请求节点数 (${NODE_COUNT})"
    exit 1
fi

declare -a NODE_IPS
for i in $(seq 0 $(( NODE_COUNT - 1 ))); do
    NODE_IP=$(echo "${NODE_ENTRIES[$i]}" | cut -d: -f1)
    NODE_IPS+=("${NODE_IP}")
done

echo "============================================"
echo "  DiffusionRL SGLang ${LOGPROB_SOURCE} ${PLACEMENT_MODE} — NEW-DESIGN 多节点"
echo "  Experiment: ${EXPERIMENT}"
echo "  集群: ${NODE_COUNT} 节点 x ${RANK_PER_NODE} GPU = $(( NODE_COUNT * RANK_PER_NODE )) 总 GPU"
echo "  Logprob mode: ${LOGPROB_SOURCE}"
echo "  Head (CHIEF_IP): ${CHIEF_IP}"
echo "  参与节点:"
for i in $(seq 0 $(( NODE_COUNT - 1 ))); do
    echo "    [${i}] ${NODE_IPS[$i]}"
done
echo "  模型路径: ${PRETRAINED_MODEL}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  Weight Sync: ${WEIGHT_SYNC_DIR}"
echo "============================================"
echo ""

# ============================================================
# ★ NCCL 环境变量 (InfiniBand/RDMA)
# ============================================================
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

# ============================================================
# ★ Step 1: 清理旧 Ray + 创建输出目录
# ============================================================
echo "[Head] 清理旧 Ray..."
ray stop 2>/dev/null || true
mkdir -p "${OUTPUT_DIR}" "${WEIGHT_SYNC_DIR}"

# ============================================================
# ★ Step 2: 启动 Ray Head
# ============================================================
echo "[Head] 启动 Ray Head on ${CHIEF_IP}:${HEAD_PORT} (gpus=${RANK_PER_NODE})..."
ray start --head \
    --node-ip-address "${CHIEF_IP}" \
    --port "${HEAD_PORT}" \
    --dashboard-host 0.0.0.0 \
    --num-gpus "${RANK_PER_NODE}"
echo "[Head] Ray Head 已启动"

# ============================================================
# ★ Step 3: SSH 启动 Worker 节点
# ============================================================
echo ""
echo "[Head] 启动 $(( NODE_COUNT - 1 )) 个 Worker..."

NCCL_ENV="NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
NCCL_IB_SL=${NCCL_IB_SL} \
NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD} \
NCCL_IB_CUDA_SUPPORT=${NCCL_IB_CUDA_SUPPORT} \
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
UCX_NET_DEVICES=${UCX_NET_DEVICES} \
NCCL_IB_HCA='${NCCL_IB_HCA}' \
NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE} \
SHARP_COLL_ENABLE_SAT=${SHARP_COLL_ENABLE_SAT} \
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION} \
NCCL_IB_TC=${NCCL_IB_TC} \
NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE} \
HF_TOKEN=${HF_TOKEN:-} \
HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}"

for i in $(seq 1 $(( NODE_COUNT - 1 ))); do
    WORKER_IP="${NODE_IPS[$i]}"
    echo "  → SSH Worker[${i}] ${WORKER_IP}..."

    ssh -f -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        "${SSH_USER}@${WORKER_IP}" \
        "cd ${REPO_ROOT} && \
         export HF_TOKEN='${HF_TOKEN:-}' && \
         export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN:-}' && \
         ${NCCL_ENV} \
         nohup ray start \
             --address '${CHIEF_IP}:${HEAD_PORT}' \
             --node-ip-address '${WORKER_IP}' \
             --num-gpus ${RANK_PER_NODE} \
         > /tmp/diffusionrl_ray_worker_${i}.log 2>&1 &"

    echo "  Worker[${i}] SSH 已发送"
done

# ============================================================
# ★ Step 4: 等待所有 Worker 加入
# ============================================================
echo ""
echo "[Head] 等待所有 ${NODE_COUNT} 个节点加入 Ray 集群..."
TIMEOUT=300
ELAPSED=0
while true; do
    N=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes())); ray.shutdown()" 2>/dev/null || echo "0")
    if [ "${N}" -ge "${NODE_COUNT}" ]; then
        echo "[Head] ✅ 所有 ${NODE_COUNT} 个节点已加入！"
        break
    fi
    if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
        echo "[WARN] 超时 (${TIMEOUT}s)。当前 ${N}/${NODE_COUNT} 个节点。继续尝试..."
        break
    fi
    echo "  ${N}/${NODE_COUNT} 个节点就绪 (${ELAPSED}s/${TIMEOUT}s)..."
    sleep 10
    ELAPSED=$(( ELAPSED + 10 ))
done

echo ""
ray status 2>/dev/null || true
echo ""

# ============================================================
# ★ Step 5: 提交训练
# ============================================================
echo "[Head] 提交训练任务: python -m diffusionrl.train_new +experiment=${EXPERIMENT}"
echo ""

export RAY_ADDRESS="${CHIEF_IP}:${HEAD_PORT}"

cd "${REPO_ROOT}"
python -m diffusionrl.train_new \
    +experiment="${EXPERIMENT}" \
    "$@"
