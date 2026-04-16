#!/bin/bash
# =============================================================================
# Debug script for GRPO train-inference consistency analysis (SD3.5)
#
# This script runs a minimal GRPO training loop with tensor dumping enabled
# to diagnose ratio drift (new_log_prob != old_log_prob on on-policy step).
#
# Debug tensors are saved to:
#   ${DEBUG_OUTPUT_DIR}/sampling/step_XXX/  -- per-step SDE tensors from sampling
#   ${DEBUG_OUTPUT_DIR}/training/step_XXX/  -- per-step SDE tensors from training
#
# After running, use scripts/analyze_debug_tensors.py to find the first
# step where sampling and training diverge.
#
# Usage:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh
#   # With CPS:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh --sampling.sde-type cps
#   # With custom eta:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh --sampling.eta 0.5
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/_check_wandb.sh"

# Default values
PRETRAINED_MODEL="stabilityai/stable-diffusion-3.5-medium"
MODEL_TYPE="sd3"
# PRETRAINED_MODEL="black-forest-labs/FLUX.1-dev"
# MODEL_TYPE="flux"
DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR:-"${REPO_ROOT}/debug_output"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/datasets/pickscore/train.txt"}
NUM_GPUS=${NUM_GPUS:-8}

# ── Batch geometry (5 core knobs — see _batch_config.sh for docs) ──
NUM_INFERENCE_STEPS=10
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
SAMPLING_FORWARD_BATCH=${SAMPLING_FORWARD_BATCH:-64}    # per-device peak forward batch during sampling
TRAINING_FORWARD_BATCH=${TRAINING_FORWARD_BATCH:-8}     # per-device peak forward batch during training
NUM_UPDATES=${NUM_UPDATES:-1}                           # gradient update steps per local batch

source "${REPO_ROOT}/scripts/_batch_config.sh"
resolve_batch_params
validate_batch_params
print_batch_params

# Parse --sampling.sde-type and --sampling.eta from cmdline for logging
SDE_TYPE_OVERRIDE=""
ETA_OVERRIDE=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--sampling.sde-type" ]; then
        SDE_TYPE_OVERRIDE="$arg"
    fi
    if [ "$prev" = "--sampling.eta" ]; then
        ETA_OVERRIDE="$arg"
    fi
    prev="$arg"
done

echo "============================================"
echo " GRPO Train-Inference Consistency Debug"
echo "============================================"
echo " DEBUG_OUTPUT_DIR: ${DEBUG_OUTPUT_DIR}"
echo " NUM_GPUS:         ${NUM_GPUS}"
echo " PROMPTS_PER_BATCH: ${PROMPTS_PER_BATCH}"
echo " NUM_SAMPLES_PER_PROMPT: ${NUM_SAMPLES_PER_PROMPT}"
echo " SDE_TYPE override: ${SDE_TYPE_OVERRIDE:-default(sde)}"
echo " ETA override:      ${ETA_OVERRIDE:-default(0.7)}"
echo "============================================"

# Clean previous debug output
rm -rf "${DEBUG_OUTPUT_DIR}"
mkdir -p "${DEBUG_OUTPUT_DIR}"

REWARD_NAME="pickscore"
REWARD_DEVICE="cuda"

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

check_wandb_auth

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type ${MODEL_TYPE} \
    --sampling.sampler-dotpath diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-components ${REWARD_NAME} \
    --reward.local-reward-device ${REWARD_DEVICE} \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 4.5 \
    --algorithm.rollout-scheduler.timestep-fraction 0.1,0.5 \
    \
    --algorithm.shuffle-seed 42 \
    --algorithm.shuffle-samples false \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --training.micro-batch-size "${MICRO_BATCH_SIZE}" \
    --training.num-updates-per-batch ${NUM_UPDATES_PER_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.kwarg clip_range=1e-4 \
    --algorithm.kwarg use_kl_penalty=true \
    --algorithm.kwarg kl_coef=0.04 \
    --algorithm.adv-normalization group \
    --algorithm.use-global-std true \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --rollout.mode direct_sampling \
        --sync.protocol disabled \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
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
    --rollout.num-rollout 1 \
    --rollout.save-steps 0 \
    --evaluation.eval-steps 0 \
    --logging.logging-steps 1 \
    --rollout.output-dir "${DEBUG_OUTPUT_DIR}/train_output" \
    --logging.report-to-wandb false \
    \
    --debug.output-dir "${DEBUG_OUTPUT_DIR}" \
    \
    "$@"

echo ""
echo "============================================"
echo " Debug tensors saved to: ${DEBUG_OUTPUT_DIR}"
echo "============================================"
