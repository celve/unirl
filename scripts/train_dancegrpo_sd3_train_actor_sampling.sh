#!/bin/bash
# =============================================================================
# DanceGRPO training with SD3 model (Training-actor sampling mode)
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
# This script runs DanceGRPO with SD3 using training-actor sampling.
# Rollout actors are disabled; sampling happens on training actors directly.
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
#   bash train_dancegrpo_sd3_train_actor_sampling.sh
#   bash train_dancegrpo_sd3_train_actor_sampling.sh --num-rollout 100 --batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
# SD3.5-medium is much smaller than FLUX, so we can use larger batch sizes
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_sd3_train_sampling"}
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
    --training-actor-direct-sampling true \
    --colocate-rollout-training true \
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
    \
    --height 512 \
    --width 512 \
    \
    --num-rollout 300 \
    --save-steps 40 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
