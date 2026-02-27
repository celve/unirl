#!/bin/bash
# =============================================================================
# DanceGRPO training with FLUX model (Separate mode - rollout/training on different GPUs)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/jwhj/DanceGRPO)
#   Script:  scripts/finetune/finetune_flux_grpo_8gpus_lora.sh
#   Command: bash scripts/finetune/finetune_flux_grpo_8gpus_lora.sh
#
# LoRA: ✅ 默认使用较小 LoRA (rank=16, alpha=32) 以降低显存占用
#        如需对标 DanceGRPO (rank=128, alpha=256)，请通过环境变量覆盖
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DanceGRPO with FLUX in separate mode where rollout and
# training actors run on different GPUs simultaneously.
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
# - guidance_scale=0.0 (no CFG during training)
# - NO KL penalty (kl_coeff=0 in original)
# - max_grad_norm=1.0 (not 2.0)
#
# NOTE: DanceGRPO originally supports multi-reward weighting (hps/clip/image_reward/pick_score).
#       diffusionrl's LocalRewardWorker only supports single reward. For multi-reward,
#       implement a custom reward worker.
#
# Usage:
#   bash train_dancegrpo_flux_separate.sh
#   bash train_dancegrpo_flux_separate.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/flux"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
# Memory-optimized settings for 4 (rollout) + 4 (training) GPUs
# FLUX is very memory hungry - use minimal settings
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type flux \
    --sampler-path diffusionrl.samplers.fsdp.flux_sampler.FluxSampler \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type flux_dance \
    --eta 0.3 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 0.0 \
    --timestep-fraction 0.6 \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 1 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --use-lora true \
    --use-fsdp true \
    \
    --height 256 \
    --width 256 \
    \
    --num-rollout 300 \
    --save-steps 40 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
