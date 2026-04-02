#!/bin/bash
# =============================================================================
# DiffusionNFT training with SD3 model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DiffusionNFT (https://github.com/NVIDIA/DiffusionNFT)
#   Script:  scripts/train_nft_sd3.py
#   Config:  config/nft.py -> geneval_sd3() or general_ocr_sd3()
#   Command: accelerate launch scripts/train_nft_sd3.py --config config/nft.py:general_ocr_sd3
#
# LoRA: The original setup uses LoRA by default (rank=32, alpha=64), matching this script
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_rollout * samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DiffusionNFT (Negative Fine-Tuning) with SD3 for OCR task (default).
# NFT uses forward diffusion in the loss function, so it doesn't require
# trajectories or log probabilities during sampling.
#
# Reference: DiffusionNFT/config/nft.py, DiffusionNFT/scripts/train_nft_sd3.py
#
# Key alignment with original DiffusionNFT (OCR task):
# - algorithm_type=nft (forward process diffusion RL)
# - beta=1.0 (interpolation weight: positive_prediction = beta*new + (1-beta)*old)
# - kl_coef=0.0001 (KL regularization coefficient, separate from beta)
# - num_inference_steps=10 (training steps, NOT 40)
# - guidance_scale=1.0 (no CFG during training)
# - periodic eval: 40 steps, default adapter, deterministic flow solver
# - adv_normalization_scope=group
# - algorithm_kwargs.train_timestep_mode=all (DiffusionNFT uses full timestep schedule)
# - adv_mode=raw
# - EMA decay: warmup curve (decay_type=2 in original)
#   - ema_flat_steps=75, ema_uprate=0.0075, ema_uphold=0.999
#
# Two beta parameters in DiffusionNFT (IMPORTANT!):
# 1. config.beta (algorithm kwargs JSON): Controls positive_prediction interpolation
#    - OCR: 0.1 (mostly use old adapter prediction)
# 2. config.train.beta (algorithm_kwargs.kl_coef): KL regularization weight
#    - Fixed: 0.0001
#
# NOTE: diffusionrl now supports dpm2 deterministic sampling for SD3 NFT path.
#
# Training-actor sampling now reuses the main driver rollout pipeline -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_nft_sd3_train_actor_sampling.sh
#   bash train_nft_sd3_train_actor_sampling.sh --rollout.num-rollout 100 --training.micro-batch-size 2
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
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/nft_sd3_ocr_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy_16.json"}
NUM_GPUS=${NUM_GPUS:-8}
TRAINING_NUM_NODES=${TRAINING_NUM_NODES:-1}
TRAINING_GPUS_PER_NODE=${TRAINING_GPUS_PER_NODE:-${NUM_GPUS}}
TOTAL_GPUS=$(( TRAINING_NUM_NODES * TRAINING_GPUS_PER_NODE ))
RAY_ADDRESS=${RAY_ADDRESS:-}
RAY_PLACEMENT_STRATEGY=${RAY_PLACEMENT_STRATEGY:-SPREAD}
WEIGHT_SYNC_DIR=${WEIGHT_SYNC_DIR:-}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE-3}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-24}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-nft_sd3_train_actor_sampling}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-ocr}
REWARD_LOCATION=${REWARD_LOCATION:-sampling_actor}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cpu}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-${ROLLOUT_TOTAL_SAMPLES}}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
NFT_ALGO_KWARG_ARGS=(
    --algorithm.kwarg "beta=0.1"
    --algorithm.kwarg "adv_mode=raw"
    --algorithm.kwarg "adv_clip_max=5.0"
    --algorithm.kwarg "use_adaptive_weight=true"
    --algorithm.kwarg "train_timestep_mode=all"
    --algorithm.kwarg "shuffle_train_timesteps=true"
    --algorithm.kwarg "apply_time_shift_in_loss=false"
    --algorithm.kwarg "use_reference_ema=true"
    --algorithm.kwarg "reference_update_timing=rollout_end"
    --algorithm.kwarg "ema_decay=0.001"
    --algorithm.kwarg "decay_type=warmup"
    --algorithm.kwarg "ema_flat_steps=75"
    --algorithm.kwarg "ema_uprate=0.0075"
    --algorithm.kwarg "ema_uphold=0.999"
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "kl_coef=0.0001"
    --algorithm.adv-normalization "group"
    --algorithm.use-global-std "true"
    --algorithm.adv-norm-eps "1e-4"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.micro-batch-size "${MICRO_BATCH_SIZE}")
fi

if [ ! -d "${PRETRAINED_MODEL}" ] && [ -d "${REPO_ROOT}/${PRETRAINED_MODEL}" ]; then
    PRETRAINED_MODEL="${REPO_ROOT}/${PRETRAINED_MODEL}"
fi
RAY_ADDRESS_ARGS=()
if [ -n "${RAY_ADDRESS}" ]; then
    RAY_ADDRESS_ARGS+=(--ray.ray-address "${RAY_ADDRESS}")
fi
SYNC_DIR_ARGS=()
if [ -n "${WEIGHT_SYNC_DIR}" ]; then
    SYNC_DIR_ARGS+=(--sync.dir "${WEIGHT_SYNC_DIR}")
fi

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-dotpath diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward.reward-components "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.shift 3.0 \
    --sampling.sde-type dpm2 \
    --sampling.num-inference-steps 10 \
    --algorithm.training-scheduler.timestep-fraction 0.99 \
    --sampling.guidance-scale 1.0 \
    --sampling.sampling-adapter old \
    "${NFT_ALGO_KWARG_ARGS[@]}" \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.mode direct_sampling \
--sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    "${RAY_ADDRESS_ARGS[@]}" \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-nodes ${TRAINING_NUM_NODES} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS_PER_NODE} \
    --ray.placement-strategy ${RAY_PLACEMENT_STRATEGY} \
    "${SYNC_DIR_ARGS[@]}" \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    --training.use-gradient-checkpointing false \
    \
    --sampling.height 512 \
    --sampling.width 512 \
    \
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --evaluation.eval-steps 60 \
    --evaluation.num-inference-steps 40 \
    --evaluation.sampling-adapter default \
    --evaluation.sde-type flow \
    --evaluation.eta 0.0 \
    --logging.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --logging.report-to-wandb ${REPORT_TO_WANDB} \
    --logging.project-name "${WANDB_PROJECT_NAME}" \
    --logging.run-name "${WANDB_RUN_NAME}" \
    --sync.protocol disabled \
    "$@"
