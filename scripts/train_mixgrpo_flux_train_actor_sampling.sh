#!/bin/bash
# =============================================================================
# MixGRPO training with FLUX model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: MixGRPO (https://github.com/HuskyKingdom/MixGRPO)
#   Script:  scripts/finetune/finetune_flux_grpo_MixGRPO.sh
#   Command: bash scripts/finetune/finetune_flux_grpo_MixGRPO.sh
#
# LoRA: ⚠️ 原版 MixGRPO 不使用 LoRA，本脚本使用 LoRA (rank=64, alpha=128) 以节省显存
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs MixGRPO with FLUX using training-actor sampling. MixGRPO uses:
# - Mixed ODE/SDE sampling with window scheduler
# - Standard SDE formulation
# - Group-based advantage normalization
#
# LoRA NOTE:
# - This script uses LoRA (rank=64, alpha=128) for memory efficiency.
# - Original MixGRPO does NOT use LoRA - it uses FSDP full fine-tuning.
# - For fair comparison with original MixGRPO, you may need to:
#   1. Disable LoRA (--use-lora false) and use more GPUs
#   2. Or modify MixGRPO to use LoRA for comparison
# - Using LoRA here enables running on fewer GPUs (8x A100-40GB).
#
# Key alignment with original MixGRPO:
# - sde_type=flux_flow (MixGRPO FLUX flow-SDE formulation)
# - eta=0.7 (noise coefficient)
# - shift=3.0 (FLUX time shift)
# - num_inference_steps=25
# - guidance_scale=3.5 (MixGRPO FLUX guidance)
# - mixed_sampling=true with sde_ratio=0.5 (50% SDE, 50% ODE)
# - Window scheduler: progressive with group_size=4, iters_per_group=25
# - NO KL penalty (kl_coeff=0.0 in original)
#
# NOTE: The following MixGRPO features are NOT implemented in diffusionrl:
# - --trimmed_ratio (outlier removal from advantage calculation)
# - --init_same_noise (same initial noise for same prompt)
# - --ignore_last (ignore last timestep)
# - --frozen_init_timesteps (freeze initial timesteps)
# - Multi-reward weighting (hps/clip/image_reward/pick_score)
#
# This is an APPROXIMATE reproduction. See plan for details.
#
# Usage:
#   bash train_mixgrpo_flux_train_actor_sampling.sh
#   bash train_mixgrpo_flux_train_actor_sampling.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/flux"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_flux_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
WINDOW_MAX_ITERS_PER_GROUP=${WINDOW_MAX_ITERS_PER_GROUP:-10}
WINDOW_MIN_ITERS_PER_GROUP=${WINDOW_MIN_ITERS_PER_GROUP:-1}
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
    --algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type flux_flow \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 3.5 \
    \
    --mixed-sampling true \
    --sde-ratio 0.5 \
    --timestep-strategy window \
    --window-strategy progressive \
    --window-group-size 4 \
    --window-iters-per-group 25 \
    --window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP} \
    --window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP} \
    --window-overlap true \
    --window-roll-back true \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    --reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --training-actor-direct-sampling true \
    --colocate-rollout-training false \
    --rollout-num-nodes 0 \
    --rollout-num-gpus-per-node 0 \
    --training-num-gpus-per-node ${NUM_GPUS} \
    --offload false \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 3 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --use-lora true \
    \
    --height 720 \
    --width 720 \
    \
    --num-rollout 300 \
    --save-steps 50 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
