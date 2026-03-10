#!/bin/bash
# =============================================================================
# MixGRPO training with FLUX model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: MixGRPO (https://github.com/HuskyKingdom/MixGRPO)
#   Script:  scripts/finetune/finetune_flux_grpo_MixGRPO.sh
#   Command: bash scripts/finetune/finetune_flux_grpo_MixGRPO.sh
#
# LoRA: Original MixGRPO does not use LoRA; this script uses LoRA (rank=64, alpha=128) to reduce memory usage
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs MixGRPO with FLUX using training-actor sampling. MixGRPO uses:
# - Mixed ODE/SDE sampling with window scheduler
# - Standard SDE formulation
# - Group-based advantage normalization
#
# LoRA NOTE:
# - This script uses LoRA (rank=64, alpha=128) for memory efficiency.
# - Original MixGRPO does NOT use LoRA - it uses FSDP full fine-tuning.
# - For fair comparison with original MixGRPO, you may need to:
#   1. Disable LoRA (--training.use-lora false) and use more GPUs
#   2. Or modify MixGRPO to use LoRA for comparison
# - Using LoRA here enables running on fewer GPUs (8x A100-40GB).
#
# Key alignment with original MixGRPO:
# - sde_type=flux_flow (MixGRPO FLUX flow-SDE formulation)
# - eta=0.7 (noise coefficient)
# - shift=3.0 (FLUX time shift)
# - num_inference_steps=25
# - guidance_scale=3.5 (MixGRPO FLUX guidance)
# - mixed_sampling=true with sde_ratio=0.5 (50% SDE, 50% ODE)
# - Window scheduler: progressive with group_size=4, iters_per_group=25
# - NO KL penalty (kl_coeff=0.0 in original)
#
# NOTE:
# - Core MixGRPO knobs used here are implemented in diffusionrl
#   (window scheduler / sde_ratio / trimmed_ratio / ignore_last / frozen_init_timesteps).
# - This script remains an approximate reproduction because reward stack,
#   dataset, and model initialization may differ from the upstream project.
#
# Training-actor sampling now reuses the main manager -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_mixgrpo_flux_train_actor_sampling.sh
#   bash train_mixgrpo_flux_train_actor_sampling.sh --rollout.num-rollout 100 --training.gradient-accumulation-batch-size 2 --training.multi-update-batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_flux_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-12}
GRADIENT_ACCUMULATION_BATCH_SIZE=${GRADIENT_ACCUMULATION_BATCH_SIZE-4}
MULTI_UPDATE_BATCH_SIZE=${MULTI_UPDATE_BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
WINDOW_MAX_ITERS_PER_GROUP=${WINDOW_MAX_ITERS_PER_GROUP:-10}
WINDOW_MIN_ITERS_PER_GROUP=${WINDOW_MIN_ITERS_PER_GROUP:-1}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_flux_train_actor_sampling}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-ocr}
REWARD_EXECUTION_MODE=${REWARD_EXECUTION_MODE:-rollout}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cpu}
if [ $(( NUM_GPUS * LOCAL_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: NUM_GPUS*LOCAL_BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( NUM_GPUS * LOCAL_BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
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
    --algorithm.algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-execution-mode "${REWARD_EXECUTION_MODE}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flux_flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 3.5 \
    \
    --sampling.sde-ratio 0.5 \
    --algorithm.window.timestep-strategy window \
    --algorithm.window.window-strategy progressive \
    --algorithm.window.window-group-size 4 \
    --algorithm.window.window-iters-per-group 25 \
    --algorithm.window.window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP} \
    --algorithm.window.window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP} \
    --algorithm.window.window-overlap true \
    --algorithm.window.window-roll-back true \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    "${GRADIENT_ACCUMULATION_ARGS[@]}" \
    --training.multi-update-batch-size ${MULTI_UPDATE_BATCH_SIZE} \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.advantage-type group \
    --algorithm.advantage-clip-max 5.0 \
    --reward.reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --sampling.training-actor-direct-sampling true \
    --sampling.direct-sampling-batch-size ${DIRECT_SAMPLING_BATCH_SIZE} \
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
    --training.lora-rank 64 \
    --training.lora-alpha 128 \
    --training.use-lora true \
    \
    --height 720 \
    --width 720 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 50 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    "$@"
