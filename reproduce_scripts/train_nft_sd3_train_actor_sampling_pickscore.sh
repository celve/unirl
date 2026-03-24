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
# - adv_normalization=group
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
# Training-actor sampling now reuses the main manager -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_nft_sd3_train_actor_sampling.sh
#   bash train_nft_sd3_train_actor_sampling.sh --rollout.control.num-rollout 100 --training.local-micro-batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"stabilityai/stable-diffusion-3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/datasets/pickscore/train.txt"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"${REPO_ROOT}/data/datasets/pickscore/test.txt"}

NUM_GPUS=${NUM_GPUS:-8}

# Rollout setttings
NUM_INFERENCE_STEPS=10 # denoising steps during rollout (sampling) stage.
NUM_SAMPLES_PER_PROMPT=24 # group size
PROMPTS_PER_BATCH=48 # number of (unique) prompts per epoch
DIRECT_SAMPLING_BATCH_SIZE=192 # Actual peak forward batch size during sampling stage.

# Training settings
LOCAL_MICRO_BATCH_SIZE=12 # Local peak forward/backward batch size during optimization
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))
NUM_UPDATES_PER_LOCAL_BATCH=${NUM_UPDATES_PER_LOCAL_BATCH:-1}

SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${LOCAL_MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.local-micro-batch-size "${LOCAL_MICRO_BATCH_SIZE}")
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH file not found: ${DATA_PATH}"
    exit 1
fi

REPORT_TO_WANDB=true
WANDB_PROJECT_NAME="diffusionrl-diffusionNFT"
WANDB_RUN_NAME="SD3.5-DiffusionNFT" # change to your own name
WANDB_LOG_MEDIA=true
WANDB_MEDIA_MAX_ITEMS=48 # Max number of image reports per logging step
WANDB_TAGS="reproduce,sd3.5,nft,pickscore" # reward=ocr, flow
WANDB_ENTITY=${WANDB_ENTITY:-"diffusionrl-reproduce"} # Set empty to skip: WANDB_ENTITY=""
LOGGING_STEPS=1

WANDB_ENTITY_ARGS=()
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARGS+=(--rollout.logging.wandb-entity "${WANDB_ENTITY}")
fi

REWARD_NAME="pickscore" # pickscore, ocr, clip, hpsv2
REWARD_DEVICE="cuda"
REWARD_LOCATION="sampling_actor" # run reward scorer on sampling actors

# Eval EMA settings (smoothed weights for stable evaluation)
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-false}
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
NFT_ALGO_KWARG_ARGS=(
    --algorithm.kwarg "beta=1"
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
    --algorithm.kwarg "ema_flat_steps=0"
    --algorithm.kwarg "ema_uprate=0.001"
    --algorithm.kwarg "ema_uphold=0.5"
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

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward.reward-model-name "${REWARD_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    --eval-data-path "${EVAL_DATA_PATH}" \
    \
    --sampling.shift 3.0 \
    --sampling.eta 0.0 \
    --sampling.sde-type dpm2 \
    --sampling.timestep-fraction 0.99 \
    --sampling.num-inference-steps ${NUM_INFERENCE_STEPS} \
    --sampling.guidance-scale 1.0 \
    --sampling.sampling-adapter old \
    "${NFT_ALGO_KWARG_ARGS[@]}" \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --training.num-updates-per-local-batch ${NUM_UPDATES_PER_LOCAL_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.topology.mode direct_rollout \
--sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    --training.use-gradient-checkpointing false \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.control.num-rollout 1000 \
    --rollout.artifacts.save-steps 0 \
    --rollout.evaluation.eval-steps 60 \
    --rollout.evaluation.num-inference-steps 40 \
    --rollout.evaluation.sampling-adapter default \
    --rollout.evaluation.sde-type flow \
    --rollout.evaluation.eta 0.0 \
    --rollout.logging.logging-steps ${LOGGING_STEPS} \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --rollout.logging.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.logging.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.logging.run-name "${WANDB_RUN_NAME}" \
    --rollout.logging.wandb-log-media ${WANDB_LOG_MEDIA} \
    --rollout.logging.wandb-media-max-items ${WANDB_MEDIA_MAX_ITEMS} \
    --rollout.logging.wandb-tags "${WANDB_TAGS}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    --sync.protocol disabled \
    "$@"
