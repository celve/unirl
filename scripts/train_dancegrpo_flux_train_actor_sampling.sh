#!/bin/bash
# =============================================================================
# DanceGRPO training with FLUX model (Training-actor sampling mode)
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
# This script runs DanceGRPO with FLUX using training-actor sampling.
# Rollout actors are disabled; sampling happens on training actors directly.
#
# Resource scheduling:
# - rollout actors disabled (training-only sampling)
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
# - guidance_scale=3.5 (DanceGRPO FLUX guidance)
# - NO KL penalty (kl_coeff=0 in original)
# - max_grad_norm=1.0 (not 2.0)
#
# NOTE: DanceGRPO originally supports multi-reward weighting (hps/clip/image_reward/pick_score).
#       diffusionrl's LocalRewardWorker only supports single reward. For multi-reward,
#       implement a custom reward worker.
#
# Usage:
#   bash train_dancegrpo_flux_train_actor_sampling.sh
#   bash train_dancegrpo_flux_train_actor_sampling.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
# Memory-optimized for 8x40GB GPUs (FLUX is ~12B params)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/flux"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_flux_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
if [ $(( NUM_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: NUM_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( NUM_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
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
    --guidance-scale 3.5 \
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
    --training-actor-direct-sampling true \
    --colocate-rollout-training false \
    --rollout-num-nodes 0 \
    --rollout-num-gpus-per-node 0 \
    --training-num-gpus-per-node ${NUM_GPUS} \
    --offload false \
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
    --fsdp-cpu-offload true \
    \
    --height 256 \
    --width 256 \
    \
    --num-rollout 300 \
    --save-steps 40 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
