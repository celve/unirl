#!/bin/bash
# =============================================================================
# Wan 2.2-I2V-A14B GRPO smoke - trainside (direct-sampling) on 1x8.
#
# NEW-DESIGN path: train_new.py + NewTrainActorGroup. I2V variant of
# scripts/run_wan22_t2v_14b_smoke.sh — image conditioning travels purely
# through the VAE-latent channel concat (transformer.config.image_dim == 0
# on WAN 2.2 14B I2V, so the CLIP vision tower is skipped).
#
# Scope: ``expand_timesteps=False`` path only (standard WAN 2.2 14B I2V).
# WAN 2.2 5B I2V (``expand_timesteps=True`` with mask blending) is out of
# scope for this smoke.
#
# Built on top of conf/experiment/grpo_wan22_i2v_new.yaml.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate the project venv (system python may lack hydra/diffusers).
VENV_DIR="${VENV_DIR:-/root/diffusionrl/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: venv not found at ${VENV_DIR}; using system python: $(which python)"
fi

# Model path (local Wan2.2-I2V-A14B-Diffusers under public_models).
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-/apdcephfs_zwfy8/share_305110755/hunyuan/public_models/Wan-AI/Wan2.2-I2V-A14B-Diffusers}"

# I2V prompt data with condition image MediaRefs.
export DATA_PATH="${DATA_PATH:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/data/datasets/sharegpt4o_image_mini/prompts.jsonl}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_PATH}}"

# Output.
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/wan22_i2v_14b_smoke}"
mkdir -p "${OUTPUT_DIR}"

# Runtime env.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/diffusionRL/models/local/.hf_home}"

echo "=========================================="
echo " Wan 2.2-I2V-14B smoke - trainside 1x8"
echo "=========================================="
echo "Model:  ${PRETRAINED_MODEL}"
echo "Data:   ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Pre-flight: WAN 2.2 14B I2V has the dual-transformer layout
# (transformer / transformer_2). Image cross-KV is folded into the VAE
# channel concat — no separate image_encoder/ subfolder expected here.
for sub in transformer transformer_2 vae text_encoder tokenizer scheduler model_index.json; do
    if [ ! -e "${PRETRAINED_MODEL}/${sub}" ]; then
        echo "ERROR: ${PRETRAINED_MODEL}/${sub} missing - is this really the Wan2.2-I2V-A14B-Diffusers checkpoint?"
        exit 1
    fi
done

# GPU snapshot.
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Stop any stale Ray cluster (train_new.py will start its own).
ray stop >/dev/null 2>&1 || true
sleep 1

# Hydra overrides.
HYDRA_OVERRIDES=(
    "run.data_path=${DATA_PATH}"
    "run.eval_data_path=${EVAL_DATA_PATH}"
    "resume.output_dir=${OUTPUT_DIR}"
    "run.num_rollouts=200"

    "sampling.height=480"
    "sampling.width=640"
    "sampling.num_frames=9"
    "sampling.num_inference_steps=10"

    # Smoke-sized batch geometry.
    "algorithm.prompts_per_rollout=8"
    "algorithm.samples_per_prompt=4"
    "training.plan.global_batch_size=32"
    "training.plan.local_batch_size=4"
    "training.plan.local_mini_batch_size=2"
    "training.plan.micro_batch_size=1"

    "rollout.plan.forward_batch_size=1"

    # LoRA: drop rank to 32.
    "training.policies.0.rank=32"
    "training.policies.0.alpha=64"

    "logging.report_to_wandb=false"
)

# Append any user-supplied overrides AFTER the smoke defaults so they win.
HYDRA_OVERRIDES+=("$@")

CMD=(
    python -m diffusionrl.train_new
    "+experiment=grpo_wan22_i2v_new"
    "${HYDRA_OVERRIDES[@]}"
)

echo "Command:"
echo "  ${CMD[*]}"
echo ""

cleanup() {
    echo ""
    echo "Cleaning up..."
    ray stop >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

echo "========== Starting training =========="
cd "${REPO_ROOT}"
exec "${CMD[@]}"
