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
# LoRA: ✅ 原版默认使用 LoRA (rank=32, alpha=64)，与本脚本一致
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DiffusionNFT (Negative Fine-Tuning) with SD3 for OCR task (default).
# NFT uses forward diffusion in the loss function, so it doesn't require
# trajectories or log probabilities during sampling.
#
# Reference: DiffusionNFT/config/nft.py, DiffusionNFT/scripts/train_nft_sd3.py
#
# Key alignment with original DiffusionNFT (OCR task):
# - loss_type=nft (forward process diffusion RL)
# - beta=1.0 (interpolation weight: positive_prediction = beta*new + (1-beta)*old)
# - kl_coef=0.0001 (KL regularization coefficient, separate from beta)
# - num_inference_steps=10 (training steps, NOT 40)
# - guidance_scale=1.0 (no CFG during training)
# - advantage_type=per_prompt (per-prompt statistic tracking)
# - loss_kwargs.nft_timestep_mode=all (DiffusionNFT uses full timestep schedule)
# - adv_mode=raw
# - EMA decay: warmup curve (decay_type=2 in original)
#   - ema_flat_steps=75, ema_uprate=0.0075, ema_uphold=0.999
#
# Two beta parameters in DiffusionNFT (IMPORTANT!):
# 1. config.beta (algorithm/loss kwargs JSON): Controls positive_prediction interpolation
#    - OCR: 0.1 (mostly use old adapter prediction)
# 2. config.train.beta (--kl-coef): KL regularization weight
#    - Fixed: 0.0001
#
# NOTE: diffusionrl now supports dpm2 deterministic sampling for SD3 NFT path.
#
# Usage:
#   bash train_nft_sd3_train_actor_sampling.sh
#   bash train_nft_sd3_train_actor_sampling.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/nft_sd3_ocr_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-3}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-24}
if [ $(( NUM_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: NUM_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( NUM_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}
NFT_ALGO_KWARGS=${NFT_ALGO_KWARGS:-'{"beta":0.1,"adv_mode":"raw","adv_clip_max":5.0,"use_adaptive_weight":true,"ema_decay":0.001}'}
NFT_LOSS_KWARGS=${NFT_LOSS_KWARGS:-'{"beta":0.1,"adv_mode":"raw","adv_clip_max":5.0,"use_adaptive_weight":true,"nft_timestep_mode":"all","nft_shuffle_timesteps":true,"nft_apply_shift":false,"use_ema":true,"ema_decay":0.001,"decay_type":"warmup","ema_flat_steps":75,"ema_uprate":0.0075,"ema_uphold":0.999}'}

if [ ! -d "${PRETRAINED_MODEL}" ] && [ -d "${REPO_ROOT}/${PRETRAINED_MODEL}" ]; then
    PRETRAINED_MODEL="${REPO_ROOT}/${PRETRAINED_MODEL}"
fi
if [ ! -d "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: PRETRAINED_MODEL path not found: ${PRETRAINED_MODEL}"
    echo "Hint: set PRETRAINED_MODEL to a local SD3 model directory."
    exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH file not found: ${DATA_PATH}"
    exit 1
fi

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm-path diffusionrl.algorithms.nft.NFTAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --loss-type nft \
    --shift 3.0 \
    --sde-type dpm2 \
    --num-inference-steps 10 \
    --guidance-scale 1.0 \
    --sampling-adapter old \
    --algorithm-kwargs-json "${NFT_ALGO_KWARGS}" \
    --loss-kwargs-json "${NFT_LOSS_KWARGS}" \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --kl-coef 0.0001 \
    --advantage-type per_prompt \
    --use-per-prompt-stat-tracker true \
    --per-prompt-buffer-size 10000 \
    \
    --training-actor-direct-sampling true \
    --colocate-rollout-training true \
    --rollout-num-nodes 0 \
    --rollout-num-gpus-per-node 0 \
    --training-num-gpus-per-node ${NUM_GPUS} \
    --offload false \
    \
    --learning-rate 3e-4 \
    --gradient-accumulation-steps auto \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --gradient-steps-per-epoch 1 \
    --max-grad-norm 1.0 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --use-lora true \
    --use-gradient-checkpointing false \
    \
    --height 512 \
    --width 512 \
    \
    --num-rollout 1000 \
    --save-steps 60 \
    --eval-steps 60 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
