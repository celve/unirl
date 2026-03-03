#!/bin/bash
# =============================================================================
# FlowGRPO training with SD3 model (Separate mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: flow_grpo (https://github.com/yifan123/flow_grpo)
#   Script:  scripts/train_sd3.py
#   Config:  config/grpo.py -> general_ocr_sd3() or general_ocr_sd3_4gpu()
#   Command: accelerate launch scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
#
# LoRA: ✅ 原版默认使用 LoRA (rank=32, alpha=64)，与本脚本一致
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs FlowGRPO with SD3 in separate mode. FlowGRPO uses:
# - SDE (standard) or CPS (Coefficient-Preserving Sampling) SDE type
# - Per-prompt advantage normalization
# - KL coefficient for stability (β=0.04)
#
# Reference: flow_grpo/config/grpo.py
#
# Key alignment with original flow_grpo:
# - sde_type=sde (default; use cps for fast variants like geneval_sd3_fast_nocfg)
# - eta=0.7 (noise coefficient)
# - shift=3.0 (SD3 time shift)
# - num_inference_steps=10 (training steps)
# - guidance_scale=4.5
# - kl_coef=0.04 (KL penalty)
# - advantage_type=per_prompt (per-prompt statistic tracking)
# - learning_rate=3e-4 (higher than DanceGRPO's 1e-5)
# - LoRA: rank=32, alpha=64 (SD3 default uses LoRA)
# - timestep_fraction=0.99 (nearly all timesteps)
#
# Batch/group configuration (8 GPU):
# - batch_size=3, num_samples_per_prompt=24 (original _get_config defaults)
# - For 4 GPU: batch_size=8, num_samples_per_prompt=16
#
# NOTE: Use --sampling.sde-type cps for CPS variant (e.g., geneval_sd3_fast_nocfg)
#
# Usage:
#   bash train_flowgrpo_sd3_separate.sh
#   bash train_flowgrpo_sd3_separate.sh --sampling.sde-type cps  # For CPS variant
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
# Memory-optimized settings for 4x40GB GPUs
# Original flow_grpo uses num_image_per_prompt=8 for 1 GPU config
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type sde \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 10 \
    --sampling.guidance-scale 4.5 \
    --sampling.timestep-fraction 0.99 \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    --training.batch-size ${BATCH_SIZE} \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.use-global-std true \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty true \
    --algorithm.kl-coef 0.04 \
    --algorithm.advantage-type per_prompt \
    --algorithm.per-prompt-buffer-size 10000 \
    \
    --ray.colocate-rollout-training false \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 3e-4 \
    --training.gradient-accumulation-steps auto \
    --training.num-inner-epochs ${NUM_INNER_EPOCHS} \
    --training.gradient-steps-per-epoch 2 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 16 \
    --training.lora-alpha 32 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --rollout.eval-steps 60 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
