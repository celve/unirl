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

export SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/home/aiops/wanghn/mmgrpo/sglang/python}"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"models/local/hunyuan-video"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/dancegrpo_hunyuan_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/video_prompts_toy.txt"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}

NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-${ROLLOUT_GPUS}}

BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-480}
NUM_FRAMES=${NUM_FRAMES:-53}
FPS=${FPS:-8}

REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-"hpsv2"}
REWARD_PATH=${REWARD_PATH:-"diffusionrl.reward.local.LocalRewardWorker"}

TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}

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

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type hunyuan \
    --sampler-engine-type sglang \
    --sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --tp-size ${TP_SIZE} \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path "${REWARD_PATH}" \
    --reward-model-name "${REWARD_MODEL_NAME}" \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type dance \
    --eta 0.25 \
    --shift 5.0 \
    --num-inference-steps 16 \
    --guidance-scale 6018.0 \
    --timestep-fraction 0.6 \
    --init-same-noise true \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy PACK \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --use-gradient-checkpointing true \
    \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --fps ${FPS} \
    \
    --num-rollout 202 \
    --save-steps 50 \
    --logging-steps 1 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
