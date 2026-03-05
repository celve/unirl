#!/bin/bash
# =============================================================================
# MixGRPO training with SD3 model (Training-actor sampling mode)
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
# This script runs MixGRPO with SD3 using training-actor sampling. MixGRPO uses:
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
# NOTE:
# - Core MixGRPO knobs used here are implemented in diffusionrl
#   (window scheduler / sde_ratio / trimmed_ratio / ignore_last / frozen_init_timesteps).
# - This script is still an adapted SD3 variant (not a bit-for-bit upstream run).
#
# Usage:
#   bash train_mixgrpo_sd3_train_actor_sampling.sh
#   bash train_mixgrpo_sd3_train_actor_sampling.sh --rollout.num-rollout 100 --training.batch-size 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Default values (can be overridden via command line)
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
WINDOW_MAX_ITERS_PER_GROUP=${WINDOW_MAX_ITERS_PER_GROUP:-10}
WINDOW_MIN_ITERS_PER_GROUP=${WINDOW_MIN_ITERS_PER_GROUP:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_sd3_train_actor_sampling}
if [ "${NUM_SAMPLES_PER_PROMPT}" -lt 2 ]; then
    echo "ERROR: MixGRPO uses group advantages; set NUM_SAMPLES_PER_PROMPT >= 2 to avoid NaN."
    exit 1
fi
if [ $(( NUM_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: NUM_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( NUM_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type sde \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 4.5 \
    \
    --sampling.sde-ratio 0.5 \
    --algorithm.window.timestep-strategy window \
    --algorithm.window.window-strategy progressive \
    --algorithm.window.window-group-size 4 \
    --algorithm.window.window-iters-per-group 25 \
    --algorithm.window.window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP} \
    --algorithm.window.window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP} \
    --algorithm.window.window-overlap true \
    --algorithm.window.window-roll-back true \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    --training.batch-size ${BATCH_SIZE} \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.advantage-type group \
    --algorithm.advantage-clip-max 5.0 \
    --reward.reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --sampling.training-actor-direct-sampling true \
    --ray.colocate-rollout-training true \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    --ray.offload false \
    \
    --training.learning-rate 1e-5 \
    --training.gradient-accumulation-steps 2 \
    --training.num-inner-epochs ${NUM_INNER_EPOCHS} \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank ${LORA_RANK} \
    --training.lora-alpha ${LORA_ALPHA} \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 50 \
    --rollout.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --rollout.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.run-name "${WANDB_RUN_NAME}" \
    "$@"
