#!/bin/bash
# =============================================================================
# DiffusionNFT SD3 debug — forward-process (trainside direct-sampling)
# Recipe: conf/experiment/debug_nft.yaml (inherits nft_sd3 + 3-rollout overrides)
#
# NEW-DESIGN path (train_new.py + NewRolloutActorGroup + NewTrainActorGroup).
# 1-node × 8-GPU layout; all train actors also sample.
#
# Usage:
#   bash scripts/debug_nft.sh                       # default
#   bash scripts/debug_nft.sh run.num_rollouts=5    # extra Hydra overrides
#   DRY_RUN=1 bash scripts/debug_nft.sh             # print command only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Activate the project venv ──
VENV_DIR="${VENV_DIR:-/root/diffusionrl/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: venv not found at ${VENV_DIR}; using system python: $(which python)"
fi

# ── Model + data paths ──
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-${REPO_ROOT}/models/local/sd3.5-medium}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/train.txt}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/datasets/pickscore/test.txt}"

# ── Output ──
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug_nft_sd3}"

# ── Forward batch size during direct-sampling on each train actor ──
export FBS="${FBS:-32}"

# ── Wandb disabled in debug mode by default ──
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ── NCCL tuning for FSDP all-reduce ──
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

echo "========================================"
echo " DiffusionNFT SD3.5 — debug (3 rollouts)"
echo "========================================"
echo "Model:   ${PRETRAINED_MODEL}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "FBS:     ${FBS}"
echo ""

# GPU status
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Stop any stale Ray cluster
ray stop >/dev/null 2>&1 || true
sleep 1

mkdir -p "${OUTPUT_DIR}"

EXPERIMENT="debug_nft"
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

cleanup() {
    echo ""
    echo "Cleaning up..."
    ray stop >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

echo ""
echo "========== Starting training =========="
cd "${REPO_ROOT}"
exec "${CMD[@]}"
