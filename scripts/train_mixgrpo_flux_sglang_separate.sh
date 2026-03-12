#!/bin/bash
# =============================================================================
# MixGRPO training with FLUX model - SGLang backend (Separate mode)
# =============================================================================
#
# Usage:
#   bash train_mixgrpo_flux_sglang_separate.sh
#   ROLLOUT_GPUS=2 TRAINING_GPUS=2 bash train_mixgrpo_flux_sglang_separate.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer local sibling sglang checkout when available; otherwise use installed package.
SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-${REPO_ROOT}/../sglang/python}"
if [ -d "${SGLANG_PYTHON_PATH}" ]; then
    export SGLANG_PYTHON_PATH
    export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"
    echo "[SGLang] Using local source: ${SGLANG_PYTHON_PATH}"
else
    echo "[SGLang] Local source not found at ${SGLANG_PYTHON_PATH}; using installed sglang."
fi


## ---- Default values (can be overridden via command line) ----
# Model & data & output configurations
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_flux_sglang_sampling"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-diffusionrl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mixgrpo_flux_sglang_sampling}

# GPU allocation
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}

# Rollout
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-32}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
NUM_UPDATE_STEPS_PER_ROLLOUT=${NUM_UPDATE_STEPS_PER_ROLLOUT:-4}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

if [ "${NUM_SAMPLES_PER_PROMPT}" -lt 2 ]; then
    echo "ERROR: MixGRPO uses group advantages; set NUM_SAMPLES_PER_PROMPT >= 2 to avoid NaN."
    exit 1
fi

# Rollout (sglang engine)
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}
REPLAY_SAMPLER_PATH=${REPLAY_SAMPLER_PATH:-diffusionrl.samplers.fsdp.flux_sampler.FluxSampler}

# Reward
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-hpsv2}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Training
UPDATE_MODE=${UPDATE_MODE:-multi_update}
if [ $(( ROLLOUT_TOTAL_SAMPLES % TRAINING_GPUS )) -ne 0 ]; then
    echo "ERROR: PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT must be divisible by TRAINING_GPUS"
    exit 1
fi
GRADIENT_ACCUMULATION_BATCH_SIZE=${GRADIENT_ACCUMULATION_BATCH_SIZE-4}
LOCAL_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / TRAINING_GPUS ))
MULTI_UPDATE_BATCH_SIZE=$(( LOCAL_BATCH_SIZE / NUM_UPDATE_STEPS_PER_ROLLOUT ))


python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type flux \
    --sampling.sampler-engine-type sglang \
    --sampling.sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --sampling.replay-log-probs "${REPLAY_LOG_PROBS}" \
    --sampling.replay-sampler-path "${REPLAY_SAMPLER_PATH}" \
    --sampling.tp-size ${TP_SIZE} \
    --algorithm.algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flux_flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 3.5 \
    \
    --algorithm.loss-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    --sampling.sde-ratio 0.16 \
    --algorithm.window.timestep-strategy window \
    --algorithm.window.window-strategy progressive \
    --algorithm.window.window-group-size 4 \
    --algorithm.window.window-iters-per-group 25 \
    --algorithm.window.window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP:-10} \
    --algorithm.window.window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP:-1} \
    --algorithm.window.window-overlap true \
    --algorithm.window.window-roll-back true \
    \
    --algorithm.prompts-per-batch ${PROMPTS_PER_BATCH} \
    --algorithm.num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.advantage-type group \
    --algorithm.advantage-clip-max 5.0 \
    --reward.reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --ray.colocate-rollout-training false \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 1e-5 \
    --training.update-mode ${UPDATE_MODE} \
    --training.gradient-accumulation-batch-size ${GRADIENT_ACCUMULATION_BATCH_SIZE} \
    --training.multi-update-batch-size ${MULTI_UPDATE_BATCH_SIZE} \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank 64 \
    --training.lora-alpha 128 \
    --training.use-lora true \
    \
    --height 720 \
    --width 720 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 50 \
    --rollout.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
