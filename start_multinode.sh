#!/bin/bash
# =============================================================================
# 多节点一键启动脚本 v2
#
# 用法：
#   bash start_multinode.sh <node_count> <rank_per_node> [-- diffusionrl overrides...]
#
# 参数说明：
#   node_count    — 要启动的节点数（从 NODE_IP_LIST 中取前 N 个）
#   rank_per_node — 每个节点的 GPU 数（通常为 8）
#   -- 之后      — 透传给 diffusionrl.train 的额外 CLI 参数（覆盖脚本默认值）
#
# 示例：
#   bash start_multinode.sh 1 4      # 1 个 node，4 个 rank
#   bash start_multinode.sh 2 8      # 2 个 node，每个 8 个 rank
#   bash start_multinode.sh 1 4 -- --rollout.mode separate   # 改用 separate 模式
#   bash start_multinode.sh 2 8 -- --rollout.mode colocate --sampling.sde-type flow
#
# 前置条件：
#   1. 环境变量 NODE_IP_LIST 存在，格式：IP:GPUS,IP:GPUS,IP:GPUS
#   2. 环境变量 CHIEF_IP 存在（Head 节点 IP）
#   3. Head 节点能免密 SSH 到所有 Worker 节点
#   4. 所有节点的 DiffusionRL 仓库在同一路径
# =============================================================================

set -euo pipefail

# ============================================================
# ★ 解析命令行参数
# ============================================================
if [ $# -lt 2 ]; then
    cat <<EOF
用法: bash start_multinode.sh <node_count> <rank_per_node> [-- extra args...]

示例:
  bash start_multinode.sh 1 4                                # 单节点，4 个 GPU
  bash start_multinode.sh 2 8                                # 2 节点，每个 8 个 GPU
  bash start_multinode.sh 1 4 -- --rollout.mode separate     # 覆盖 rollout 模式
  bash start_multinode.sh 2 8 -- --sampling.sde-type flow    # 覆盖任意训练参数

环境变量检查:
  CHIEF_IP=${CHIEF_IP:-未设置}
  NODE_IP_LIST=${NODE_IP_LIST:-未设置}

EOF
    exit 1
fi

NODE_COUNT=$1
RANK_PER_NODE=$2
shift 2

# 解析 -- 分隔符后的额外参数
EXTRA_ARGS=()
if [ "$#" -gt 0 ]; then
    if [ "$1" = "--" ]; then
        shift
    fi
    EXTRA_ARGS=("$@")
fi

# ============================================================
# ★ 验证环境变量
# ============================================================
if [ -z "${CHIEF_IP:-}" ]; then
    echo "[ERROR] 环境变量 CHIEF_IP 未设置"
    exit 1
fi

if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "[ERROR] 环境变量 NODE_IP_LIST 未设置"
    exit 1
fi

# ============================================================
# ★ 从 NODE_IP_LIST 中解析节点 IP 列表
# ============================================================
# NODE_IP_LIST 格式: IP:GPUS,IP:GPUS,IP:GPUS
# 例如: 28.48.2.84:8,28.49.19.220:8

IFS=',' read -ra NODE_ENTRIES <<< "${NODE_IP_LIST}"

if [ ${#NODE_ENTRIES[@]} -lt ${NODE_COUNT} ]; then
    echo "[ERROR] NODE_IP_LIST 中的节点数 (${#NODE_ENTRIES[@]}) 少于请求节点数 (${NODE_COUNT})"
    exit 1
fi

# 提取前 NODE_COUNT 个节点的 IP 和 GPU 数
declare -a NODE_IPS
declare -a NODE_GPUS

for i in $(seq 0 $(( NODE_COUNT - 1 ))); do
    ENTRY="${NODE_ENTRIES[$i]}"
    # 格式: IP:GPUS
    NODE_IP=$(echo "${ENTRY}" | cut -d: -f1)
    NODE_GPU=$(echo "${ENTRY}" | cut -d: -f2)
    NODE_IPS+=("${NODE_IP}")
    NODE_GPUS+=("${NODE_GPU}")
done

# ============================================================
# ★ 用户配置区 — 按实际情况修改
# ============================================================
SSH_USER="${SSH_USER:-$(whoami)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# 训练参数（与 start.sh 保持一致）
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var must be set}"
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
WANDB_ENTITY="${WANDB_ENTITY:-qianqiu95-personal}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-train_grpo_sd3_multinode}"
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR:-/apdcephfs_nj10/share_301739632/qianqiu/hy-exploration/diffusionrl_weight_sync/flowgrpo_fast_sd3_multinode}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_grpo_sd3_multinode.sh}"

HEAD_PORT="${HEAD_PORT:-6379}"
# ============================================================

echo "============================================"
echo "  DiffusionRL 多节点启动"
echo "  启动模式: ${NODE_COUNT} 节点 x ${RANK_PER_NODE} GPU"
echo "  总 GPU 数: $(( NODE_COUNT * RANK_PER_NODE ))"
echo "  Head 节点 (CHIEF_IP): ${CHIEF_IP}"
echo "  参与节点:"
for i in $(seq 0 $(( NODE_COUNT - 1 ))); do
    echo "    [${i}] ${NODE_IPS[$i]} (GPU: ${NODE_GPUS[$i]})"
done
echo "  训练脚本: ${TRAIN_SCRIPT}"
echo "  仓库路径: ${REPO_ROOT}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
echo "  额外参数: ${EXTRA_ARGS[*]}"
fi
echo "============================================"
echo ""

# ============================================================
# ★ Step 1: 清理本机旧 Ray 进程
# ============================================================
echo "[Head] 清理本机 Ray 进程..."
ray stop >/dev/null 2>&1 || true

# ============================================================
# ★ Step 2: 多节点时，先启动 Ray Head，再 SSH Worker
# ============================================================
TRAIN_ROLE="auto"
if [ ${NODE_COUNT} -gt 1 ]; then
    echo ""
    echo "[Head] 多节点模式：先设置 NCCL 环境变量..."
    # NCCL 环境变量需要在 ray start 之前 export，
    # 这样 Ray daemon 和所有 worker 进程都会继承
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

    echo "[Head] 启动 Ray Head..."
    GPUS_PER_NODE="${RANK_PER_NODE}"
    ray start --head \
        --node-ip-address "${CHIEF_IP}" \
        --port "${HEAD_PORT}" \
        --dashboard-host 0.0.0.0 \
        --num-gpus "${GPUS_PER_NODE}"
    echo "[Head] Ray Head 已启动: ${CHIEF_IP}:${HEAD_PORT}"

    echo ""
    echo "[Head] 启动 $(( NODE_COUNT - 1 )) 个 Worker 节点..."

    for i in $(seq 1 $(( NODE_COUNT - 1 ))); do
        WORKER_IP="${NODE_IPS[$i]}"
        WORKER_GPU="${NODE_GPUS[$i]}"
        echo "  → SSH 到 ${SSH_USER}@${WORKER_IP} (INDEX=${i}, GPU=${WORKER_GPU})"

        # 通过 SSH 在 worker 上后台启动（-f 让 SSH 立即返回）
        ssh -f -o StrictHostKeyChecking=no \
            -o ConnectTimeout=10 \
            -o BatchMode=yes \
            "${SSH_USER}@${WORKER_IP}" \
            "cd ${REPO_ROOT} && \
             HF_TOKEN='${HF_TOKEN}' \
             HUGGINGFACE_HUB_TOKEN='${HUGGINGFACE_HUB_TOKEN}' \
             WANDB_ENTITY='${WANDB_ENTITY}' \
             WEIGHT_SYNC_DIR='${WEIGHT_SYNC_DIR}' \
             INDEX=${i} \
             CHIEF_IP='${CHIEF_IP}' \
             LOCAL_IP='${WORKER_IP}' \
             HOST_NUM=${NODE_COUNT} \
             HOST_GPU_NUM=${RANK_PER_NODE} \
             NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-48} \
             PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-24} \
             ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-24} \
             ROLLOUT_GPUS_PER_NODE=${RANK_PER_NODE} \
             TRAINING_GPUS_PER_NODE=${RANK_PER_NODE} \
             TRAINING_FORWARD_BATCH=${TRAINING_FORWARD_BATCH:-12} \
             NUM_UPDATES=${NUM_UPDATES:-2} \
             nohup bash reproduce_scripts/${TRAIN_SCRIPT} auto \
             > /tmp/diffusionrl_worker_${i}.log 2>&1 &"
        echo "  Worker ${i} SSH 已发送"
    done

    echo "[Head] 所有 Worker SSH 命令已发送，等待节点加入 (30s)..."
    sleep 30

    # Head 已手动启动 Ray，训练脚本只需 train 角色
    TRAIN_ROLE="train"
fi

# ============================================================
# ★ Step 3: 在本机（Head 节点）启动训练任务
# ============================================================
echo ""
echo "[Head] 启动训练任务 (role=${TRAIN_ROLE})..."
echo ""

cd "${REPO_ROOT}"

export WANDB_API_KEY="${WANDB_API_KEY:?ERROR: WANDB_API_KEY env var must be set}"
HF_TOKEN="${HF_TOKEN}" \
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_DIR}" \
HOST_NUM="${NODE_COUNT}" \
HOST_GPU_NUM="${RANK_PER_NODE}" \
CHIEF_IP="${CHIEF_IP}" \
LOCAL_IP="${CHIEF_IP}" \
INDEX=0 \
NUM_SAMPLES_PER_PROMPT=48 \
ROLLOUT_GPUS_PER_NODE=${RANK_PER_NODE} \
PROMPTS_PER_BATCH=24 \
ROLLOUT_BATCH_SIZE=24 \
TRAINING_GPUS_PER_NODE=${RANK_PER_NODE} \
TRAINING_FORWARD_BATCH=12 \
NUM_UPDATES=2 \
bash "reproduce_scripts/${TRAIN_SCRIPT}" "${TRAIN_ROLE}" "${EXTRA_ARGS[@]}"
