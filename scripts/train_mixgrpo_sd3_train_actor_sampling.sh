#!/bin/bash
# =============================================================================
# MixGRPO training with SD3 model (Training-actor sampling mode)
# =============================================================================
#
# REPRODUCE TARGET:
#   Project: MixGRPO (https://github.com/Tencent-Hunyuan/MixGRPO)
#   Algorithm: MixGRPO with SD3 model (adapted from FLUX version)
#
# NOTE: MixGRPO paper mentions SD3.5-M LoRA experiments for comparison but doesn't
#       elaborate on the hyper-parameters or open-source the code for SD training.
#       This script uses MixGRPO's configurations for FLUX.
#       SD3 is much smaller than FLUX (~2B vs ~12B), making it ideal for testing.
#
# LoRA: Use LoRA (rank=32, alpha=64) to reduce memory usage
#
# =============================================================================
# batch_geometry: total_samples = prompts_per_rollout * samples_per_prompt
# per_rank_batch = total_samples / num_train_gpus (must be divisible)
#
# This script runs MixGRPO with SD3 using training-actor sampling. MixGRPO uses:
# - Mixed ODE/SDE sampling with window scheduler
# - Standard SDE formulation
# - Group-based advantage normalization
#
# Key parameters (adapted from MixGRPO FLUX for SD3):
# - sde_type=flow (standard SDE for SD3)
# - eta=0.7 (noise coefficient, same as MixGRPO FLUX)
# - shift=3.0 (SD3 time shift)
# - num_inference_steps=25
# - guidance_scale=4.5 (SD3 benefits from CFG, unlike FLUX)
# - mixed_sampling=true with sde_ratio=0.16 (16% SDE, 84% ODE)
# - Window scheduler: progressive with group_size=4, iters_per_group=25
# - NO KL penalty (same as MixGRPO)
#
# NOTE:
# - Core MixGRPO knobs used here are implemented in diffusionrl
#   (window scheduler / sde_ratio / trimmed_ratio / skip_last_timestep / skip_initial_timesteps).
# - This script is still an adapted SD3 variant (not a bit-for-bit upstream run).
#
# Training-actor sampling now reuses the main driver rollout pipeline -> rollout_buffer -> train path.
# The main speed knob left in this branch is rollout-side reward execution.
#
# Usage:
#   bash train_mixgrpo_sd3_train_actor_sampling.sh
#   bash train_mixgrpo_sd3_train_actor_sampling.sh --rollout.control.num-rollout 100 --training.local-micro-batch-size 2 --training.num-updates-per-local-batch 2
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment variables (.env)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi


## ---- Default configurations (can be overridden via command line) ----
# Model & data & output
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_sd3_train_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_sd3_train_actor_sampling}

# GPU allocation
NUM_GPUS=${NUM_GPUS:-8}

# Rollout
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-32}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

# Rollout (direct sampling)
DIRECT_SAMPLING_BATCH_SIZE=${DIRECT_SAMPLING_BATCH_SIZE:-${ROLLOUT_TOTAL_SAMPLES}}

# Reward
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward}
REWARD_LOCATION=${REWARD_LOCATION:-sampling_actor}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-hpsv2}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cuda}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
MIXGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=false"
    --algorithm.adv-normalization "group"
    --algorithm.adv-clip-abs "5.0"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
    --algorithm.component-mix-stage "${REWARD_MIX_MODE}"
)

# Training
NUM_UPDATES_PER_LOCAL_BATCH=${NUM_UPDATES_PER_LOCAL_BATCH:-4}
LOCAL_MICRO_BATCH_SIZE=${LOCAL_MICRO_BATCH_SIZE:-4}
LOCAL_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / NUM_GPUS ))


python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --sampling.sampler-path diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler \
    --algorithm.algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --reward.reward-location "${REWARD_LOCATION}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 4.5 \
    \
    "${MIXGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.rollout-scheduler.timestep-strategy window \
    --algorithm.rollout-scheduler.window-strategy progressive \
    --algorithm.rollout-scheduler.window-group-size 4 \
    --algorithm.rollout-scheduler.window-iters-per-group 25 \
    --algorithm.rollout-scheduler.window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP:-10} \
    --algorithm.rollout-scheduler.window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP:-1} \
    --algorithm.rollout-scheduler.window-overlap true \
    --algorithm.rollout-scheduler.window-roll-back true \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --rollout.topology.mode direct_rollout \
--sampling.max-samples-per-request ${DIRECT_SAMPLING_BATCH_SIZE} \
    --ray.rollout-num-nodes 0 \
    --ray.rollout-num-gpus-per-node 0 \
    --ray.training-num-gpus-per-node ${NUM_GPUS} \
    \
    --training.learning-rate 1e-5 \
    --training.num-updates-per-local-batch ${NUM_UPDATES_PER_LOCAL_BATCH} \
    --training.local-micro-batch-size ${LOCAL_MICRO_BATCH_SIZE} \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.control.num-rollout 300 \
    --rollout.artifacts.save-steps 50 \
    --rollout.logging.logging-steps 1 \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --rollout.logging.report-to-wandb ${REPORT_TO_WANDB} \
    --rollout.logging.project-name "${WANDB_PROJECT_NAME}" \
    --rollout.logging.run-name "${WANDB_RUN_NAME}" \
    --sync.protocol disabled \
    "$@"
