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
# LoRA: ✅ 原版默认使用 LoRA (rank=32, alpha=64)，与本脚本一致
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs flow_grpo with SD3 using training-actor sampling. flow_grpo uses:
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
# NOTE: Use --sde-type cps for CPS variant (e.g., geneval_sd3_fast_nocfg)
#
# Usage:
#   bash train_flowgrpo_sd3_train_actor_sampling.sh
#   bash train_flowgrpo_sd3_train_actor_sampling.sh --sde-type cps  # For CPS variant
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
# Memory-optimized settings for 8x40GB GPUs with offload
# Reduced settings to avoid OOM with SD3's CFG (2x forward)
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
if [ $(( NUM_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: NUM_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( NUM_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}
NUM_INFERENCE_STEPS_OVERRIDE=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--num-inference-steps" ]; then
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

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps ${NUM_INFERENCE_STEPS} \
    --guidance-scale 4.5 \
    --timestep-fraction 0.99 \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --use-global-std true \
    --clip-range 1e-4 \
    --use-kl-penalty true \
    --kl-coef 0.04 \
    --advantage-type per_prompt \
    --per-prompt-buffer-size 10000 \
    \
    --training-actor-direct-sampling true \
    --colocate-rollout-training false \
    --rollout-num-nodes 0 \
    --rollout-num-gpus-per-node 0 \
    --training-num-gpus-per-node ${NUM_GPUS} \
    --offload false \
    \
    --learning-rate 3e-4 \
    --gradient-accumulation-steps auto \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --gradient-steps-per-epoch 2 \
    --max-grad-norm 1.0 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --use-lora true \
    --use-fsdp true \
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
