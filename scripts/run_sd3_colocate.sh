#!/bin/bash
# =============================================================================
# FlowGRPO SD3.5-medium — vllm-omni + FSDP DP=8 + colocate on 1×8 H20
#
# NEW-DESIGN path (train_new.py + NewRolloutActorGroup + NewTrainActorGroup).
# Single-node launcher: starts a local Ray cluster and submits training
# in one shot. No head/worker ceremony needed for 1 node.
#
# Layout:
#   - 8 vllm-omni rollout actors (TP=1, 1 GPU each)
#   - 8 FSDP DP=8 train actors (1 GPU each, colocated)
#   - tensor_payload weight sync (gloo gather → Ray IPC, no NCCL conflicts)
#
# Usage:
#   bash scripts/run_sd3_colocate.sh                  # default
#   bash scripts/run_sd3_colocate.sh run.num_rollouts=100  # extra overrides
#   DRY_RUN=1 bash scripts/run_sd3_colocate.sh        # print command only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Activate the project venv (the system conda python lacks hydra) ──
VENV_DIR="${VENV_DIR:-/root/diffusionrl/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: venv not found at ${VENV_DIR}; using system python: $(which python)"
fi

# ── Model path (local SD3.5-medium checkout) ──
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/models/local/sd3.5-medium}"

# ── Data paths (pickscore) ──
export DATA_PATH="${DATA_PATH:-/root/diffusionrl/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-/root/diffusionrl/data/datasets/pickscore/test.txt}"

# ── Output ──
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flowgrpo_fast_sd3_new_colocate}"

# ── Forward batch size (rollout-side) ──
export FBS="${FBS:-32}"

# ── Disable wandb by default (override with REPORT_TO_WANDB=true) ──
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ── NCCL tuning (single-node, still needed for FSDP all-reduce) ──
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# ── Ensure clean GPU state ──
echo "========================================"
echo " FlowGRPO SD3.5 — vllm-omni + colocate"
echo "========================================"
echo "Model:   ${PRETRAINED_MODEL}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "FBS:     ${FBS}"
echo ""

# Check GPU availability
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Stop any stale Ray cluster
ray stop >/dev/null 2>&1 || true
sleep 1

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ── Build the command ──
EXPERIMENT="flowgrpo_fast_sd3_new_colocate"
HYDRA_OVERRIDES=(
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
)

# Append any user-supplied overrides
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

# ── Optional GPU monitoring (background) ──
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

# ── Cleanup on exit ──
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    ray stop >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

# ── Launch ──
echo ""
echo "========== Starting training =========="
cd "${REPO_ROOT}"
exec "${CMD[@]}"
