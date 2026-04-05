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
# Current defaults in this script:
# - sde_type=flow (use cps via CLI for CPS variants)
# - eta=0.7
# - shift=3.0
# - num_inference_steps=10
# - guidance_scale=1.0
# - kl_coef=0.04
# - adv_normalization_scope=group
# - learning_rate=3e-4
# - LoRA: rank=32, alpha=64
# - timestep_fraction=0.99
# - training.num_updates_per_batch (+ optional training.micro_batch_size when tuning memory)
# - reward_location=sampling_actor
# - reward_components defaults to pickscore
# - prompts_per_rollout=16, samples_per_prompt=8 on 8 GPUs
# - sampling.max_samples_per_request only controls OOM-safe request splitting;
#   rollout_total_samples still equals prompts_per_rollout * samples_per_prompt
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
source "${REPO_ROOT}/scripts/_check_wandb.sh"


# Default values (can be overridden via env or command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}

# Rollout setttings
NUM_INFERENCE_STEPS=10
NUM_SAMPLES_PER_PROMPT=8 # group size
PROMPTS_PER_BATCH=16 # number of prompts per epoch
DIRECT_SAMPLING_BATCH_SIZE=8 # Lower peak sampling batch size to reduce OOM risk.

# Training settings
MICRO_BATCH_SIZE=2 # Lower forward/backward batch size during optimization.
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.micro-batch-size "${MICRO_BATCH_SIZE}")
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

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-flowgrpo_sd3_train_actor_sampling}
WANDB_LOG_MEDIA=${WANDB_LOG_MEDIA:-true}
WANDB_MEDIA_MAX_ITEMS=${WANDB_MEDIA_MAX_ITEMS:-16}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-pickscore}
REWARD_LOCATION=${REWARD_LOCATION:-sampling_actor}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cuda}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
FLOWGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=true"
    --algorithm.kwarg "kl_coef=0.04"
    --algorithm.adv-normalization "group"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

LOGGING_STEPS=1

check_wandb_auth

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-dotpath diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-components "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --sampling.guidance-scale 1.0 \
    --algorithm.rollout-scheduler.timestep-fraction 0.99 \
    \
    "${FLOWGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.mode direct_sampling \
--ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
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
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --evaluation.eval-steps 60 \
    --logging.logging-steps ${LOGGING_STEPS} \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --logging.report-to-wandb ${REPORT_TO_WANDB} \
    --logging.project-name "${WANDB_PROJECT_NAME}" \
    --logging.run-name "${WANDB_RUN_NAME}" \
    --logging.log-media ${WANDB_LOG_MEDIA} \
    --logging.media-max-items ${WANDB_MEDIA_MAX_ITEMS} \
    --sync.protocol disabled \
    "$@"
