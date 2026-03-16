#!/bin/bash
# =============================================================================
# DanceGRPO training with HunyuanVideo - SGLang backend (Separate mode)
# =============================================================================
#
# Usage:
#   bash train_dancegrpo_hunyuan_sglang_separate.sh
#   ROLLOUT_GPUS=4 TRAINING_GPUS=4 bash train_dancegrpo_hunyuan_sglang_separate.sh
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

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/hunyuan-video"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}

NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${ROLLOUT_GPUS}}

GRADIENT_ACCUMULATION_BATCH_SIZE=${GRADIENT_ACCUMULATION_BATCH_SIZE-2}

HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}
REWARD_PATH=${REWARD_PATH:-"diffusionrl.reward.local.LocalRewardWorker"}

TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
    echo "Set DATA_PATH to an existing prompt file (recommended: ${REPO_ROOT}/data/samples/video_prompts_toy.txt)."
    exit 1
fi

TOTAL_SAMPLES=$((PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT))
if [ $((TOTAL_SAMPLES % TRAINING_GPUS)) -ne 0 ]; then
    echo "ERROR: total_samples (${TOTAL_SAMPLES} = ${PROMPTS_PER_BATCH}x${NUM_SAMPLES_PER_PROMPT}) must be divisible by TRAINING_GPUS (${TRAINING_GPUS})"
    exit 1
fi
LOCAL_BATCH_SIZE=$((TOTAL_SAMPLES / TRAINING_GPUS))
GRADIENT_ACCUMULATION_ARGS=()
if [ -n "${GRADIENT_ACCUMULATION_BATCH_SIZE}" ]; then
    GRADIENT_ACCUMULATION_ARGS+=(--training.gradient-accumulation-batch-size "${GRADIENT_ACCUMULATION_BATCH_SIZE}")
fi

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type hunyuan \
    --sampling.sampler-engine-type sglang \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --sampling.replay-log-probs "${REPLAY_LOG_PROBS}" \
    --sampling.tp-size ${TP_SIZE} \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-path "${REWARD_PATH}" \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type dance \
    --sampling.eta 0.25 \
    --sampling.time-shift 5.0 \
    --sampling.num-inference-steps 16 \
    --sampling.guidance-scale 6018.0 \
    --sampling.timestep-fraction 0.6 \
    --sampling.init-same-noise true \
    \
    --algorithm.algorithm-kwargs "{\"shuffle_seed\":${SHUFFLE_SEED},\"shuffle_samples\":${SHUFFLE_SAMPLES}}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${GRADIENT_ACCUMULATION_ARGS[@]}" \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --algorithm.clip-range 1e-4 \
    --algorithm.use-kl-penalty false \
    --algorithm.adv-normalization group \
    --algorithm.adv-clip-abs 5.0 \
    --algorithm.eval-ema-decay ${EVAL_EMA_DECAY} \
    --algorithm.eval-ema-update-interval ${EVAL_EMA_UPDATE_INTERVAL} \
    \
    --ray.colocate-rollout-training false \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy PACK \
    \
    --training.learning-rate 1e-5 \
    --training.update-mode single_update \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.use-gradient-checkpointing true \
    \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    --rollout.num-rollout 202 \
    --rollout.save-steps 50 \
    --rollout.logging-steps 1 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    "$@"
