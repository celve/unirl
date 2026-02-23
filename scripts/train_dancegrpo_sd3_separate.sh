#!/bin/bash
# =============================================================================
# DanceGRPO training with SD3 model (Separate mode - inference/training on different GPUs)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: DanceGRPO (https://github.com/XueZeyue/DanceGRPO)
#   Algorithm: DanceGRPO with SD3 model (adapted from FLUX version)
#
# NOTE: DanceGRPO originally supports SD v1.4 and FLUX, not SD3.
#       This script adapts DanceGRPO's algorithm to SD3 for testing purposes.
#       SD3 is much smaller than FLUX (~2B vs ~12B), making it ideal for testing.
#
# LoRA: ✅ 默认使用 LoRA (rank=16, alpha=32) 以降低显存占用
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_batch * num_samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs DanceGRPO with SD3 in separate mode where inference and
# training actors run on different GPUs simultaneously.
#
# Key parameters (adapted from DanceGRPO FLUX for SD3):
# - sde_type=sde (standard SDE formulation for SD3)
# - eta=0.3 (noise coefficient, same as DanceGRPO FLUX)
# - shift=3.0 (SD3 time shift)
# - timestep_fraction=0.6 (only train on 60% of timesteps, same as DanceGRPO)
# - guidance_scale=4.5 (SD3 benefits from CFG)
# - NO KL penalty (kl_coeff=0, same as DanceGRPO)
# - max_grad_norm=1.0 (same as DanceGRPO)
#
# Usage:
#   bash train_dancegrpo_sd3_separate.sh
#   bash train_dancegrpo_sd3_separate.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_sd3_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
# SD3 is smaller, so we can use more GPUs for inference
INFERENCE_GPUS=${INFERENCE_GPUS:-4}
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
    --model-type sd3 \
    --sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.3 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 4.5 \
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
    --colocate-inference-training false \
    --inference-num-gpus-per-node ${INFERENCE_GPUS} \
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
    --height 512 \
    --width 512 \
    \
    --num-rollout 300 \
    --save-steps 40 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
