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

# Load environment variables (.env)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

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
NUM_UPDATES_PER_BATCH=${NUM_UPDATES_PER_BATCH:-4}
ROLLOUT_TOTAL_SAMPLES=$(( PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT ))

# Rollout (sglang engine)
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_SAMPLER_PATH=${REPLAY_SAMPLER_PATH:-diffusionrl.samplers.fsdp.flux_sampler.FluxSampler}

# Reward
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-hpsv2}
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
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE-4}


python -m diffusionrl.train \
    --model.pretrained-model-ckpt-path "${PRETRAINED_MODEL}" \
    --model.model-type flux \
    --rollout.mode separate \
    --rollout.rollout-engine sglang \
    --rollout.num-gpus-per-actor ${TP_SIZE} \
    --rollout.tp-size ${TP_SIZE} \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --sampling.replay-sampler-path "${REPLAY_SAMPLER_PATH}" \
    --algorithm.algorithm-dotpath diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward.reward-components "${REWARD_MODEL_NAME}" \
    --data-source-dotpath diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 25 \
    --sampling.guidance-scale 3.5 \
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
    --training.micro-batch-size ${MICRO_BATCH_SIZE} \
    --training.num-updates-per-batch ${NUM_UPDATES_PER_BATCH} \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.lora-rank 64 \
    --training.lora-alpha 128 \
    --training.use-lora true \
    \
    --sampling.height 720 \
    --sampling.width 720 \
    \
    --rollout.num-rollout 300 \
    --rollout.save-steps 50 \
    --logging.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --sync.protocol tensor_payload \
    "$@"
