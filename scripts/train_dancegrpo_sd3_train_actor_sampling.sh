#!/bin/bash
# =============================================================================
# DanceGRPO training with SD3 model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/XueZeyue/DanceGRPO)
#   Algorithm: DanceGRPO with SD3 model (adapted from FLUX version)
#
# NOTE: DanceGRPO originally supports SD v1.4 and FLUX, not SD3.
#       This script adapts DanceGRPO's algorithm to SD3 for testing purposes.
#       SD3 is much smaller than FLUX (~2B vs ~12B), making it ideal for testing.
#
# LoRA: Use LoRA by default (rank=16, alpha=32) to reduce memory usage
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_rollout * samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DanceGRPO with SD3 using training-actor sampling.
# Rollout actors are disabled; sampling happens on training actors directly.
#
# Key parameters (adapted from DanceGRPO FLUX for SD3):
# - sde_type=sde (standard SDE formulation for SD3)
# - eta=0.3 (noise coefficient, same as DanceGRPO FLUX)
# - shift=3.0 (SD3 time shift)
# - timestep_fraction=0.6 (only train on 60% of timesteps, same as DanceGRPO)
# - guidance_scale=4.5 (SD3 benefits from CFG)
# - NO KL penalty (kl_coeff=0, same as DanceGRPO)
# - max_grad_norm=1.0 (same as DanceGRPO)
#
# Usage:
#   bash train_dancegrpo_sd3_train_actor_sampling.sh
#   bash train_dancegrpo_sd3_train_actor_sampling.sh --rollout.num-rollout 100 --training.gradient-accumulation-batch-size 2 --training.multi-update-batch-size 2
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
# SD3.5-medium is much smaller than FLUX, so we can use larger batch sizes
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}
GRADIENT_ACCUMULATION_BATCH_SIZE=${GRADIENT_ACCUMULATION_BATCH_SIZE-1}
MULTI_UPDATE_BATCH_SIZE=${MULTI_UPDATE_BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-dancegrpo_sd3_train_actor_sampling}
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
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type sde \
    --sampling.eta 0.3 \
    --sampling.time-shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 4.5 \
    --sampling.timestep-fraction 0.6 \
    \
    --algorithm.algorithm-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${GRADIENT_ACCUMULATION_ARGS[@]}" \
    --training.multi-update-batch-size ${MULTI_UPDATE_BATCH_SIZE} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.adv-normalization group \
    --algorithm.adv-clip-abs 5.0 \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --sampling.sampling-mode training_actor \
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
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 40 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    "$@"
