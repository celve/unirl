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

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE-2}

HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}

TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}
# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
DANCEGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=false"
    --algorithm.adv-normalization "group"
    --algorithm.adv-clip-abs "5.0"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
    echo "Set DATA_PATH to an existing prompt file (recommended: ${REPO_ROOT}/data/samples/video_prompts_toy.txt)."
    exit 1
fi

TOTAL_SAMPLES=$((PROMPTS_PER_BATCH * NUM_SAMPLES_PER_PROMPT))
LOCAL_BATCH_SIZE=$((TOTAL_SAMPLES / TRAINING_GPUS))
LOCAL_MICRO_BATCH_ARGS=()
if [ -n "${MICRO_BATCH_SIZE}" ]; then
    LOCAL_MICRO_BATCH_ARGS+=(--training.micro-batch-size "${MICRO_BATCH_SIZE}")
fi

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type hunyuan \
    --rollout.topology.mode separate \
    --rollout.topology.service-engine sglang \
    --rollout.topology.service-num-gpus ${TP_SIZE} \
    --rollout.topology.engine-tp-size ${TP_SIZE} \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-model-name "${REWARD_MODEL_NAME}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type dance \
    --sampling.eta 0.25 \
    --sampling.shift 5.0 \
    --sampling.num-inference-steps 16 \
    --sampling.guidance-scale 6018.0 \
    --algorithm.training-share-rollout-indices false \
    --algorithm.rollout-scheduler.timestep-fraction 0.99 \
    --algorithm.training-scheduler.timestep-fraction 0.6 \
    --sampling.init-same-noise true \
    \
    "${DANCEGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    "${LOCAL_MICRO_BATCH_ARGS[@]}" \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy PACK \
    \
    --training.learning-rate 1e-5 \
    --training.max-grad-norm 1.0 \
    --training.weight-decay 0.0001 \
    --training.use-gradient-checkpointing true \
    \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    --rollout.control.num-rollout 202 \
    --rollout.artifacts.save-steps 50 \
    --rollout.logging.logging-steps 1 \
    --rollout.artifacts.output-dir "${OUTPUT_DIR}" \
    --sync.protocol tensor_payload \
    "$@"
