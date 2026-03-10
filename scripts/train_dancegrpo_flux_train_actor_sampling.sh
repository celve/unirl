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
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
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
# - sde_type=flux_dance (DanceGRPO FLUX formulation)
# - eta=0.3 (noise coefficient)
# - shift=3.0 (FLUX time shift)
# - timestep_fraction=0.6 (only train on 60% of timesteps)
# - guidance_scale=3.5 (DanceGRPO FLUX guidance)
# - NO KL penalty (kl_coeff=0 in original)
# - max_grad_norm=1.0 (not 2.0)
#
# NOTE: DanceGRPO originally supports multi-reward weighting (hps/clip/image_reward/pick_score).
#       diffusionrl's LocalRewardWorker only supports single reward. For multi-reward,
#       implement a custom reward worker.
#
# Training-actor sampling now reuses the main manager -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_dancegrpo_flux_train_actor_sampling.sh
#   bash train_dancegrpo_flux_train_actor_sampling.sh --rollout.num-rollout 100 --training.gradient-accumulation-batch-size 2 --training.multi-update-batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
# Memory-optimized for 8x40GB GPUs (FLUX is ~12B params)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}
GRADIENT_ACCUMULATION_BATCH_SIZE=${GRADIENT_ACCUMULATION_BATCH_SIZE-1}
MULTI_UPDATE_BATCH_SIZE=${MULTI_UPDATE_BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-dancegrpo_flux_train_actor_sampling}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-ocr}
REWARD_EXECUTION_MODE=${REWARD_EXECUTION_MODE:-rollout}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cpu}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-${ROLLOUT_TOTAL_SAMPLES}}
UPDATE_MODE=${UPDATE_MODE:-multi_update}
if [ $(( DIRECT_SAMPLING_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
if [ "${DIRECT_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_TOTAL_SAMPLES}" ] && [ $(( ROLLOUT_TOTAL_SAMPLES % DIRECT_SAMPLING_BATCH_SIZE )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must evenly divide rollout_total_samples (${ROLLOUT_TOTAL_SAMPLES})"
    exit 1
fi
GRADIENT_ACCUMULATION_ARGS=()
if [ -n "${GRADIENT_ACCUMULATION_BATCH_SIZE}" ]; then
    GRADIENT_ACCUMULATION_ARGS+=(--training.gradient-accumulation-batch-size "${GRADIENT_ACCUMULATION_BATCH_SIZE}")
fi

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type flux \
    --sampling.sampler-path diffusionrl.samplers.fsdp.flux_sampler.FluxSampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-execution-mode "${REWARD_EXECUTION_MODE}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flux_dance \
    --sampling.eta 0.3 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.direct-sampling-batch-size ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 3.5 \
    --sampling.timestep-fraction 0.6 \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    "${GRADIENT_ACCUMULATION_ARGS[@]}" \
    --training.multi-update-batch-size ${MULTI_UPDATE_BATCH_SIZE} \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.advantage-type group \
    --algorithm.advantage-clip-max 5.0 \
    \
    --sampling.training-actor-direct-sampling true \
    --ray.colocate-rollout-training true \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    --ray.offload false \
    \
    --training.learning-rate 1e-5 \
    --training.update-mode ${UPDATE_MODE} \
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
