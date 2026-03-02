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

export SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/home/aiops/wanghn/mmgrpo/sglang/python}"

PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${REPO_ROOT}/models/local/flux.1-dev"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/mixgrpo_flux_sglang_separate"}
DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/samples/prompts_toy.json"}

ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAINING_GPUS=${TRAINING_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_SAMPLES_PER_PROMPT=${NUM_SAMPLES_PER_PROMPT:-12}
REWARD_MIX_MODE=${REWARD_MIX_MODE:-reward_aggr}
WINDOW_MAX_ITERS_PER_GROUP=${WINDOW_MAX_ITERS_PER_GROUP:-10}
WINDOW_MIN_ITERS_PER_GROUP=${WINDOW_MIN_ITERS_PER_GROUP:-1}
TP_SIZE=${TP_SIZE:-1}
SGLANG_LOGPROB_MODE=${SGLANG_LOGPROB_MODE:-replay}
REPLAY_LOG_PROBS=${REPLAY_LOG_PROBS:-true}

if [ "${NUM_SAMPLES_PER_PROMPT}" -lt 2 ]; then
    echo "ERROR: MixGRPO uses group advantages; set NUM_SAMPLES_PER_PROMPT >= 2 to avoid NaN."
    exit 1
fi
if [ $(( TRAINING_GPUS * BATCH_SIZE % NUM_SAMPLES_PER_PROMPT )) -ne 0 ]; then
    echo "ERROR: TRAINING_GPUS*BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT"
    exit 1
fi
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-$(( TRAINING_GPUS * BATCH_SIZE / NUM_SAMPLES_PER_PROMPT ))}
NUM_INNER_EPOCHS=${NUM_INNER_EPOCHS:-1}

python -m diffusionrl.train \
    --pretrained-model-saved-path "${PRETRAINED_MODEL}" \
    --model-type flux \
    --sampler-engine-type sglang \
    --sglang-logprob-mode "${SGLANG_LOGPROB_MODE}" \
    --replay-log-probs "${REPLAY_LOG_PROBS}" \
    --tp-size ${TP_SIZE} \
    --algorithm-path diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm \
    --reward-path diffusionrl.reward.local.LocalRewardWorker \
    --reward-model-name ocr \
    --data-source-path diffusionrl.data.data_source.ImageRLDataSource \
    --data-path "${DATA_PATH}" \
    \
    --sde-type flux_flow \
    --eta 0.7 \
    --shift 3.0 \
    --num-inference-steps 25 \
    --guidance-scale 3.5 \
    \
    --sde-ratio 0.5 \
    --timestep-strategy window \
    --window-strategy progressive \
    --window-group-size 4 \
    --window-iters-per-group 25 \
    --window-max-iters-per-group ${WINDOW_MAX_ITERS_PER_GROUP} \
    --window-min-iters-per-group ${WINDOW_MIN_ITERS_PER_GROUP} \
    --window-overlap true \
    --window-roll-back true \
    \
    --prompts-per-batch ${PROMPTS_PER_BATCH} \
    --batch-size ${BATCH_SIZE} \
    --num-samples-per-prompt ${NUM_SAMPLES_PER_PROMPT} \
    --clip-range 1e-4 \
    --use-kl-penalty false \
    --advantage-type group \
    --advantage-clip-max 5.0 \
    --reward-mix-mode ${REWARD_MIX_MODE} \
    \
    --colocate-rollout-training false \
    --rollout-num-gpus-per-node ${ROLLOUT_GPUS} \
    --training-num-gpus-per-node ${TRAINING_GPUS} \
    --placement-strategy SPREAD \
    \
    --learning-rate 1e-5 \
    --gradient-accumulation-steps 3 \
    --num-inner-epochs ${NUM_INNER_EPOCHS} \
    --max-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --use-lora true \
    \
    --height 720 \
    --width 720 \
    \
    --num-rollout 300 \
    --save-steps 50 \
    --logging-steps 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
