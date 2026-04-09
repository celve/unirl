#!/bin/bash
# =============================================================================
# MixGRPO training with SD3 model - SGLang backend (Separate mode)
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
# Key parameters (adapted from MixGRPO FLUX for SD3):
# - sde_type=flow, eta=0.7, shift=3.0
# - num_inference_steps=25, guidance_scale=4.5
# - mixed_sampling=true with sde_ratio=0.16 (16% SDE, 84% ODE)
# - Window scheduler: progressive with window_size=4, iters_per_window=25
# - NO KL penalty (same as MixGRPO)
# - LoRA rank=32, alpha=64
#
# Usage:
#   bash train_mixgrpo_sd3_sglang_separate.sh
#   ROLLOUT_GPUS=2 TRAINING_GPUS=2 bash train_mixgrpo_sd3_sglang_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/_check_wandb.sh"

# Prefer local sibling sglang checkout when available; otherwise use installed package.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi

# Model & data & output
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_sd3_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_sd3_sglang_separate}

# GPU allocation
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}

# Rollout
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}

# Reward
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-hpsv2}
LOCAL_REWARD_DEVICE=${LOCAL_REWARD_DEVICE:-cuda}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Eval EMA settings
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
NUM_UPDATES_PER_BATCH=${NUM_UPDATES_PER_BATCH:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}

check_wandb_auth

python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --rollout.mode separate \
    --rollout.rollout-engine sglang \
    --rollout.num-gpus-per-actor ${TP_SIZE} \
    --rollout.tp-size ${TP_SIZE} \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-components "${REWARD_MODEL_NAME}" \
    --reward.local-reward-device "${LOCAL_REWARD_DEVICE}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
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
    --algorithm.rollout-scheduler.window-size 4 \
    --algorithm.rollout-scheduler.iters-per-window 25 \
    --algorithm.rollout-scheduler.max-iters-per-window ${WINDOW_MAX_ITERS_PER_GROUP:-10} \
    --algorithm.rollout-scheduler.min-iters-per-window ${WINDOW_MIN_ITERS_PER_GROUP:-1} \
    --algorithm.rollout-scheduler.overlap-size 3 \
    --algorithm.rollout-scheduler.roll-back true \
    \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 1e-5 \
    --training.num-updates-per-batch ${NUM_UPDATES_PER_BATCH} \
    --training.micro-batch-size ${MICRO_BATCH_SIZE} \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --sampling.height 512 \
    --sampling.width 512 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 50 \
    --logging.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --logging.report-to-wandb ${REPORT_TO_WANDB} \
    --logging.project-name "${WANDB_PROJECT_NAME}" \
    --logging.run-name "${WANDB_RUN_NAME}" \
    --sync.protocol nccl_broadcast \
    "$@"
