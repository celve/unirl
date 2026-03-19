#!/bin/bash
# =============================================================================
# FlowGRPO training with SD3 model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: flow_grpo (https://github.com/yifan123/flow_grpo)
#   Script:  scripts/train_sd3.py
#   Config:  config/grpo.py -> general_ocr_sd3() or general_ocr_sd3_4gpu()
#   Command: accelerate launch scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
#
# LoRA: The original setup uses LoRA by default (rank=32, alpha=64), matching this script
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_rollout * samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs flow_grpo with SD3 using training-actor sampling. flow_grpo uses:
# - SDE (standard) or CPS (Coefficient-Preserving Sampling) SDE type
# - Group advantage normalization
# - KL coefficient for stability (β=0.04)
#
# Reference: flow_grpo/config/grpo.py
#
# Key alignment with original flow_grpo:
# - sde_type=flow (default; use cps for fast variants like geneval_sd3_fast_nocfg)
# - eta=0.7 (noise coefficient)
# - shift=3.0 (SD3 time shift)
# - num_inference_steps=10 (training steps)
# - guidance_scale=4.5
# - kl_coef=0.04 (KL penalty)
# - adv_normalization=group
# - learning_rate=3e-4 (higher than DanceGRPO's 1e-5)
# - LoRA: rank=32, alpha=64 (SD3 default uses LoRA)
# - timestep_fraction=0.99 (nearly all timesteps)
# - training.local_update_batch_size + num_updates_per_local_batch for Flow-style multi-update inner loops
# - sampling.direct_sampling_batch_size only controls OOM-safe request splitting;
#   rollout_total_samples still equals prompts_per_rollout * samples_per_prompt
#
# Batch/group configuration (8 GPU):
# - batch_size=3, samples_per_prompt=24 (original _get_config defaults)
# - For 4 GPU: batch_size=8, samples_per_prompt=16
#
# NOTE: Use --sampling.sde-type cps for CPS variant (e.g., geneval_sd3_fast_nocfg)
#
# Usage:
#   bash train_flowgrpo_sd3_train_actor_sampling.sh
#   bash train_flowgrpo_sd3_train_actor_sampling.sh --sampling.sde-type cps  # For CPS variant
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"stabilityai/stable-diffusion-3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/datasets/ocr/train.txt"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"${REPO_ROOT}/data/datasets/ocr/test.txt"}
NUM_GPUS=8

# Rollout setttings
NUM_INFERENCE_STEPS=10 # denoising steps during rollout (sampling) stage.
NUM_SAMPLES_PER_PROMPT=24 # group size
PROMPTS_PER_BATCH=48 # number of (unique) prompts per epoch
DIRECT_SAMPLING_BATCH_SIZE=192 # Actual peak forward batch size during sampling stage.

# Training settings
LOCAL_MICRO_BATCH_SIZE=12 # Local peak forward/backward batch size during optimization
LOCAL_UPDATE_BATCH_SIZE=72 # Local optimizer-update batch size. Set `prompts_per_rollout * samples_per_prompt` // NUM_GPUS // n for n updates.
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

if [ $(( DIRECT_SAMPLING_BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
if [ "${DIRECT_SAMPLING_BATCH_SIZE}" -lt "${ROLLOUT_TOTAL_SAMPLES}" ] && [ $(( ROLLOUT_TOTAL_SAMPLES % DIRECT_SAMPLING_BATCH_SIZE )) -ne 0 ]; then
    echo "ERROR: DIRECT_SAMPLING_BATCH_SIZE must evenly divide rollout_total_samples (${ROLLOUT_TOTAL_SAMPLES})"
    exit 1
fi
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${LOCAL_MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}")
fi
NUM_INFERENCE_STEPS_OVERRIDE=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--sampling.num-inference-steps" ]; then
        NUM_INFERENCE_STEPS_OVERRIDE="$arg"
    fi
    prev="$arg"
done
if [ -n "$NUM_INFERENCE_STEPS_OVERRIDE" ]; then
    NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS_OVERRIDE"
fi
if [ "${NUM_INFERENCE_STEPS}" -lt 2 ]; then
    echo "WARNING: num_inference_steps < 2 can lead to empty training timesteps and no optimizer step."
fi

REPORT_TO_WANDB=true
WANDB_PROJECT_NAME="diffusionrl-flowgrpo"
WANDB_RUN_NAME="SD3.5-Flow-GRPO" # change to your own name
WANDB_LOG_MEDIA=true
WANDB_MEDIA_MAX_ITEMS=48 # Max number of image reports per logging step
WANDB_TAGS="reproduce,sd3.5,flow_fast,ocr" # reward=ocr, flow_fast
WANDB_ENTITY=${WANDB_ENTITY:-"diffusionrl-reproduce"} # Set empty to skip: WANDB_ENTITY=""
LOGGING_STEPS=1

WANDB_ENTITY_ARGS=()
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARGS+=(--rollout.wandb-entity "${WANDB_ENTITY}")
fi

REWARD_NAME="ocr" # pickscore, ocr, clip, hpsv2
REWARD_DEVICE="cuda"
REWARD_LOCATION="sampling_actor" # run reward scorer on sampling actors
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-false}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}


python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-model-name ${REWARD_NAME} \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device ${REWARD_DEVICE} \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    --eval-data-path "${EVAL_DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.time-shift 3.0 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 4.5 \
    --sampling.timestep-fraction 0.1,0.2 \
    \
    --algorithm.algorithm-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --training.local-update-batch-size ${LOCAL_UPDATE_BATCH_SIZE} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty true \
    --algorithm.kl-coef 0.04 \
    --algorithm.adv-normalization group \
    --algorithm.use-global-std true \
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
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 10000 \
    --rollout.save-steps 0 \
    --rollout.eval-steps 60 \
    --rollout.logging-steps ${LOGGING_STEPS} \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    --rollout.wandb-log-media ${WANDB_LOG_MEDIA} \
    --rollout.wandb-media-max-items ${WANDB_MEDIA_MAX_ITEMS} \
    --rollout.wandb-tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    "$@"
