#!/bin/bash
# =============================================================================
# FlowGRPO training with SD3 model - SGLang backend (Separate mode)
# =============================================================================
#
# NOTE:
#   training_actor_direct_sampling=true currently requires sampler_engine_type=fsdp.
#   This script is the SGLang equivalent in separate rollout/training mode.
#
# Usage:
#   bash train_flowgrpo_sd3_sglang_separate.sh
#   bash train_flowgrpo_sd3_sglang_separate.sh --sde-type cps
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/home/aiops/wanghn/mmgrpo/sglang/python}"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/sd3.5-medium"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/flowgrpo_sd3_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-8}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}

if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type sd3 \
    --sampler-engine-type sglang \
    --sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --replay-log-probs "${REPLAY_LOG_PROBS}" \
    --tp-size ${TP_SIZE} \
    --algorithm-path diffusionrl.algorithms.grpo.GRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type sde \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps 10 \
    --guidance-scale 4.5 \
    --timestep-fraction 0.99 \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --use-global-std true \
    --clip-range 1e-4 \
    --use-kl-penalty true \
    --kl-coef 0.04 \
    --advantage-type per_prompt \
    --per-prompt-buffer-size 10000 \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 3e-4 \
    --gradient-accumulation-steps auto \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --gradient-steps-per-epoch 2 \
    --max-grad-norm 1.0 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --use-lora true \
    \
    --height 512 \
    --width 512 \
    \
    --num-rollout 1000 \
    --save-steps 60 \
    --eval-steps 60 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
