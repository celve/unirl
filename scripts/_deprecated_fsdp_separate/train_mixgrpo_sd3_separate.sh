#!/bin/bash
# =============================================================================
# MixGRPO training with SD3 model (Separate mode - rollout/training on different GPUs)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: MixGRPO (https://github.com/Tencent-Hunyuan/MixGRPO)
#   Algorithm: MixGRPO with SD3 model (adapted from FLUX version)
#
# NOTE: MixGRPO paper mentions SD3.5-M LoRA experiments for comparison.
#       This script adapts MixGRPO's algorithm to SD3 for testing purposes.
#       SD3 is much smaller than FLUX (~2B vs ~12B), making it ideal for testing.
#
# LoRA: ✅ 使用 LoRA (rank=32, alpha=64) 以节省显存
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs MixGRPO with SD3 in separate mode. MixGRPO uses:
# - Mixed ODE/SDE sampling with window scheduler
# - Standard SDE formulation
# - Group-based advantage normalization
#
# Key parameters (adapted from MixGRPO FLUX for SD3):
# - sde_type=sde (standard SDE for SD3)
# - eta=0.7 (noise coefficient, same as MixGRPO FLUX)
# - shift=3.0 (SD3 time shift)
# - num_inference_steps=25
# - guidance_scale=4.5 (SD3 benefits from CFG, unlike FLUX)
# - mixed_sampling=true with sde_ratio=0.5 (50% SDE, 50% ODE)
# - Window scheduler: progressive with group_size=4, iters_per_group=25
# - NO KL penalty (same as MixGRPO)
#
# NOTE: The following MixGRPO features are NOT implemented in diffusionrl:
# - --trimmed_ratio (outlier removal from advantage calculation)
# - --init_same_noise (same initial noise for same prompt)
# - --ignore_last (ignore last timestep)
# - --frozen_init_timesteps (freeze initial timesteps)
#
# Usage:
#   bash train_mixgrpo_sd3_separate.sh
#   bash train_mixgrpo_sd3_separate.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_sd3_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 4.5 \
    \
    --mixed-sampling true \
    --sde-ratio 0.5 \
    --timestep-strategy window \
    --window-strategy progressive \
    --window-group-size 4 \
    --window-iters-per-group 25 \
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
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 2 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --use-lora true \
    --use-fsdp true \
    \
    --height 512 \
    --width 512 \
    \
    --num-rollout 300 \
    --save-steps 50 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
