#!/bin/bash
# =============================================================================
# FlowGRPO training with SD3 model - SGLang backend (Separate mode)
# =============================================================================
#
# NOTE:
#   direct sampling now uses rollout.mode='direct_sampling' with direct_sampling only.
#   This script is the SGLang equivalent in separate rollout/training mode.
#
# Usage:
#   bash train_flowgrpo_sd3_sglang_separate.sh
#   bash train_flowgrpo_sd3_sglang_separate.sh --sampling.sde-type cps
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

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/ocr_prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_SAMPLES=${SHUFFLE_SAMPLES:-true}

# Eval EMA settings (smoothed weights for stable evaluation)
EVAL_EMA_DECAY=${EVAL_EMA_DECAY:-0.9}
EVAL_EMA_UPDATE_INTERVAL=${EVAL_EMA_UPDATE_INTERVAL:-1}
FLOWGRPO_ALGO_KWARG_ARGS=(
    --algorithm.shuffle-seed "${SHUFFLE_SEED}"
    --algorithm.shuffle-samples "${SHUFFLE_SAMPLES}"
    --algorithm.kwarg "clip_range=1e-4"
    --algorithm.kwarg "use_kl_penalty=true"
    --algorithm.kwarg "kl_coef=0.04"
    --algorithm.adv-normalization "group"
    --algorithm.eval-ema-decay "${EVAL_EMA_DECAY}"
    --algorithm.eval-ema-update-interval "${EVAL_EMA_UPDATE_INTERVAL}"
)

PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}

python -m diffusionrl.train \
    --model.pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model.model-type sd3 \
    --rollout.mode separate \
    --rollout.rollout-engine sglang \
    --rollout.num-gpus-per-actor ${TP_SIZE} \
    --rollout.tp-size ${TP_SIZE} \
    --sampling.logprob-source "${SGLANG_LOGPROB_MODE}" \
    --algorithm.algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward.reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sampling.sde-type flow \
    --sampling.eta 0.7 \
    --sampling.shift 3.0 \
    --sampling.num-inference-steps 10 \
    --sampling.guidance-scale 4.5 \
    --algorithm.rollout-scheduler.timestep-fraction 0.99 \
    \
    "${FLOWGRPO_ALGO_KWARG_ARGS[@]}" \
    --algorithm.prompts-per-rollout ${PROMPTS_PER_BATCH} \
    --training.micro-batch-size ${BATCH_SIZE} \
    --algorithm.samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    \
    --ray.rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --ray.training-num-gpus-per-node ${TRAINING_GPUS} \
    --ray.placement-strategy SPREAD \
    \
    --training.learning-rate 3e-4 \
    --training.max-grad-norm 1.0 \
    --training.lora-rank 32 \
    --training.lora-alpha 64 \
    --training.use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --rollout.num-rollout 1000 \
    --rollout.save-steps 60 \
    --evaluation.eval-steps 60 \
    --logging.logging-steps 10 \
    --rollout.output-dir "${OUTPUT_DIR}" \
    --sync.protocol tensor_payload \
    "$@"
