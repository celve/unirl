#!/bin/bash
# =============================================================================
# DanceGRPO training with FLUX model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/jwhj/DanceGRPO)
#   Script:  scripts/finetune/finetune_flux_grpo_8gpus_lora.sh
#   Command: bash scripts/finetune/finetune_flux_grpo_8gpus_lora.sh
#
# LoRA: Use a smaller LoRA by default (rank=16, alpha=32) to reduce memory usage
#       Override via environment variables to match DanceGRPO (rank=128, alpha=256)
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_rollout * samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DanceGRPO with FLUX using training-actor sampling.
# Rollout actors are disabled; sampling happens on training actors directly.
#
# Resource scheduling:
# - rollout actors disabled (training-only sampling)
#
# LoRA NOTE:
# - This script uses LoRA (default rank=16, alpha=32) for memory efficiency.
# - Original DanceGRPO has TWO versions:
#   1. finetune_flux_grpo.sh - Full fine-tuning (requires more GPUs/memory)
#   2. finetune_flux_grpo_8gpus_lora.sh - LoRA version (~20GB VRAM per GPU)
# - For fair comparison, use the LoRA version: finetune_flux_grpo_8gpus_lora.sh
#
# Key alignment with original DanceGRPO:
# - sde_type=dance (DanceGRPO FLUX formulation)
# - eta=0.3 (noise coefficient)
# - shift=3.0 (FLUX time shift)
# - timestep_fraction=0.6 (only train on 60% of timesteps)
# - guidance_scale=3.5 (DanceGRPO FLUX guidance)
# - NO KL penalty (kl_coeff=0 in original)
# - max_grad_norm=1.0 (not 2.0)
#
# NOTE: DanceGRPO originally supports multi-reward weighting (hps/clip/image_reward/pick_score).
#       diffusionrl's LocalRewardScorer only supports single reward. For multi-reward,
#       implement a custom reward scorer.
#
# Training-actor sampling now reuses the main manager -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_dancegrpo_flux_train_actor_sampling.sh
#   bash train_dancegrpo_flux_train_actor_sampling.sh --rollout.num-rollout 100 --training.local-micro-batch-size 2 --training.local-update-batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables (.env)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi


# Default values (can be overridden via command line)
# Memory-optimized for 8x40GB GPUs (FLUX is ~12B params)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}
LOCAL_MICRO_BATCH_SIZE=${LOCAL_MICRO_BATCH_SIZE-1}
LOCAL_UPDATE_BATCH_SIZE=${LOCAL_UPDATE_BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-dancegrpo_flux_train_actor_sampling}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-ocr}
REWARD_LOCATION=${REWARD_LOCATION:-sampling_actor}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cpu}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-${ROLLOUT_TOTAL_SAMPLES}}
LOCAL_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / NUM_GPUS ))
NUM_UPDATES_PER_LOCAL_BATCH=$(( LOCAL_BATCH_SIZE / LOCAL_UPDATE_BATCH_SIZE ))
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${LOCAL_MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}")
fi

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type flux \
    --sampling.sampler-path diffusionrl.samplers.fsdp.flux_sampler.FluxSampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardScorer \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type dance \
    --sampling.eta 0.3 \
    --sampling.time-shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 3.5 \
    --sampling.timestep-fraction 0.6 \
    \
    --algorithm.algorithm-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --training.local-update-batch-size ${LOCAL_UPDATE_BATCH_SIZE} \
    --training.num-updates-per-local-batch ${NUM_UPDATES_PER_LOCAL_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.adv-normalization group \
    --algorithm.adv-clip-abs 5.0 \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --rollout.mode direct_rollout \
    --rollout.service-engine fsdp \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    --ray.offload false \
    \
    --training.learning-rate 1e-5 \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank ${LORA_RANK} \
    --training.lora-alpha ${LORA_ALPHA} \
    --training.use-lora true \
    --training.fsdp-cpu-offload false \
    \
    --height 256 \
    --width 256 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 40 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    "$@"
