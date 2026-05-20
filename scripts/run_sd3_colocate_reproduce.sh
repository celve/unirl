#!/bin/bash
# =============================================================================
# FlowGRPO SD3.5-medium REPRODUCE — vllm-omni + FSDP DP=8 + colocate
#
# Reproduce-grade training on 1×8 H20:
#   48 prompts × 16 samples = 768 global batch (same as 2×8 reference)
#   8 FSDP actors with grad-accum to match reference effective gradient
#
# NEW-DESIGN path (train_new.py + NewRolloutActorGroup + NewTrainActorGroup)
#
# Usage:
#   bash scripts/run_sd3_colocate_reproduce.sh
#   bash scripts/run_sd3_colocate_reproduce.sh run.num_rollouts=100
#   DRY_RUN=1 bash scripts/run_sd3_colocate_reproduce.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Activate the project venv ──
VENV_DIR="${VENV_DIR:-/root/diffusionrl/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: venv not found at ${VENV_DIR}; using system python: $(which python)"
fi

# ── Model path ──
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/models/local/sd3.5-medium}"

# ── Data paths ──
export DATA_PATH="${DATA_PATH:-/root/diffusionrl/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-/root/diffusionrl/data/datasets/pickscore/test.txt}"

# ── Output ──
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_sd3_reproduce}"

# ── Rollout forward batch size ──
# 768 images per rollout; vllm-omni queues up to FBS at a time.
export FBS="${FBS:-48}"

# ── WandB ──
export WANDB_MODE="${WANDB_MODE:-online}"

# ── NCCL tuning ──
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

echo "=============================================="
echo " FlowGRPO SD3.5 REPRODUCE — vllm-omni colocate"
echo "=============================================="
echo "Model:   ${PRETRAINED_MODEL}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "FBS:     ${FBS}"
echo ""
echo "Reproduce config:"
echo "  prompts_per_rollout = 48"
echo "  samples_per_prompt  = 16"
echo "  global_batch_size   = 768"
echo "  actor_count         = 8 (DP=8)"
echo "  micro_batch_size    = 1 (grad-accum ×48)"
echo "  num_updates         = 2"
echo "  LoRA                = rank=32 alpha=64"
echo "  lr                  = 3e-4 constant"
echo "  EMA                 = decay=0.99 interval=2"
echo ""

# GPU status
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Stop stale Ray
ray stop >/dev/null 2>&1 || true
sleep 1

mkdir -p "${OUTPUT_DIR}"

# ── Build command ──
EXPERIMENT="flowgrpo_fast_sd3_new_colocate_reproduce"
HYDRA_OVERRIDES=(
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
)
HYDRA_OVERRIDES+=("$@")

CMD=(
    python -m diffusionrl.train_new
    "+experiment=${EXPERIMENT}"
    "${HYDRA_OVERRIDES[@]}"
)

echo "Command:"
echo "  ${CMD[*]}"
echo ""

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] Exiting without running."
    exit 0
fi

# ── GPU monitoring ──
GPU_LOG="${OUTPUT_DIR}/gpu_monitor.csv"
echo "timestamp,gpu_idx,mem_used_mib,mem_total_mib,gpu_util_pct,temp_c" > "${GPU_LOG}"
(
    while true; do
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx used total util temp; do
            echo "$(date +%Y-%m-%dT%H:%M:%S),${idx// /},${used// /},${total// /},${util// /},${temp// /}"
        done >> "${GPU_LOG}"
        sleep 10
    done
) &
GPU_MONITOR_PID=$!
echo "GPU monitor started (PID=${GPU_MONITOR_PID}, log=${GPU_LOG})"

cleanup() {
    echo ""
    echo "Cleaning up..."
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    ray stop >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

echo ""
echo "========== Starting training (reproduce) =========="
cd "${REPO_ROOT}"
exec "${CMD[@]}"
